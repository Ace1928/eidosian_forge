import base64
import json
import logging
import os
from collections import namedtuple
from typing import (
import numpy as np
import pandas as pd
from pyspark import RDD, SparkContext, cloudpickle
from pyspark.ml import Estimator, Model
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import (
from pyspark.ml.util import (
from pyspark.resource import ResourceProfileBuilder, TaskResourceRequests
from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import col, countDistinct, pandas_udf, rand, struct
from pyspark.sql.types import (
from scipy.special import expit, softmax  # pylint: disable=no-name-in-module
import xgboost
from xgboost import XGBClassifier
from xgboost.compat import is_cudf_available, is_cupy_available
from xgboost.core import Booster, _check_distributed_params
from xgboost.sklearn import DEFAULT_N_ESTIMATORS, XGBModel, _can_use_qdm
from xgboost.training import train as worker_train
from .._typing import ArrayLike
from .data import (
from .params import (
from .utils import (
class _SparkXGBEstimator(Estimator, _SparkXGBParams, MLReadable, MLWritable):
    _input_kwargs: Dict[str, Any]

    def __init__(self) -> None:
        super().__init__()
        self._set_xgb_params_default()
        self._set_fit_params_default()
        self._set_predict_params_default()
        self._setDefault(num_workers=1, device='cpu', use_gpu=False, force_repartition=False, repartition_random_shuffle=False, feature_names=None, feature_types=None, arbitrary_params_dict={})
        self.logger = get_logger(self.__class__.__name__)

    def setParams(self, **kwargs: Any) -> None:
        """
        Set params for the estimator.
        """
        _extra_params = {}
        if 'arbitrary_params_dict' in kwargs:
            raise ValueError("Invalid param name: 'arbitrary_params_dict'.")
        for k, v in kwargs.items():
            if k == self.features_cols.name:
                raise ValueError(f"Unsupported param '{k}' please use features_col instead.")
            if k in _inverse_pyspark_param_alias_map:
                raise ValueError(f'Please use param name {_inverse_pyspark_param_alias_map[k]} instead.')
            if k in _pyspark_param_alias_map:
                if k == _inverse_pyspark_param_alias_map[self.featuresCol.name] and isinstance(v, list):
                    real_k = self.features_cols.name
                    k = real_k
                else:
                    real_k = _pyspark_param_alias_map[k]
                    k = real_k
            if self.hasParam(k):
                if k == 'features_col' and isinstance(v, list):
                    self._set(**{'features_cols': v})
                else:
                    self._set(**{str(k): v})
            else:
                if k in _unsupported_xgb_params or k in _unsupported_fit_params or k in _unsupported_predict_params or (k in _unsupported_train_params):
                    err_msg = _unsupported_params_hint_message.get(k, f"Unsupported param '{k}'.")
                    raise ValueError(err_msg)
                _extra_params[k] = v
        _check_distributed_params(kwargs)
        _existing_extra_params = self.getOrDefault(self.arbitrary_params_dict)
        self._set(arbitrary_params_dict={**_existing_extra_params, **_extra_params})

    @classmethod
    def _pyspark_model_cls(cls) -> Type['_SparkXGBModel']:
        """
        Subclasses should override this method and
        returns a _SparkXGBModel subclass
        """
        raise NotImplementedError()

    def _create_pyspark_model(self, xgb_model: XGBModel) -> '_SparkXGBModel':
        return self._pyspark_model_cls()(xgb_model)

    def _convert_to_sklearn_model(self, booster: bytearray, config: str) -> XGBModel:
        xgb_sklearn_params = self._gen_xgb_params_dict(gen_xgb_sklearn_estimator_param=True)
        sklearn_model = self._xgb_cls()(**xgb_sklearn_params)
        sklearn_model.load_model(booster)
        sklearn_model._Booster.load_config(config)
        return sklearn_model

    def _query_plan_contains_valid_repartition(self, dataset: DataFrame) -> bool:
        """
        Returns true if the latest element in the logical plan is a valid repartition
        The logic plan string format is like:

        == Optimized Logical Plan ==
        Repartition 4, true
        +- LogicalRDD [features#12, label#13L], false

        i.e., the top line in the logical plan is the last operation to execute.
        so, in this method, we check the first line, if it is a "Repartition" operation,
        and the result dataframe has the same partition number with num_workers param,
        then it means the dataframe is well repartitioned and we don't need to
        repartition the dataframe again.
        """
        num_partitions = dataset.rdd.getNumPartitions()
        assert dataset._sc._jvm is not None
        query_plan = dataset._sc._jvm.PythonSQLUtils.explainString(dataset._jdf.queryExecution(), 'extended')
        start = query_plan.index('== Optimized Logical Plan ==')
        start += len('== Optimized Logical Plan ==') + 1
        num_workers = self.getOrDefault(self.num_workers)
        if query_plan[start:start + len('Repartition')] == 'Repartition' and num_workers == num_partitions:
            return True
        return False

    def _repartition_needed(self, dataset: DataFrame) -> bool:
        """
        We repartition the dataset if the number of workers is not equal to the number of
        partitions. There is also a check to make sure there was "active partitioning"
        where either Round Robin or Hash partitioning was actively used before this stage.
        """
        if self.getOrDefault(self.force_repartition):
            return True
        try:
            if self._query_plan_contains_valid_repartition(dataset):
                return False
        except Exception:
            pass
        return True

    def _get_distributed_train_params(self, dataset: DataFrame) -> Dict[str, Any]:
        """
        This just gets the configuration params for distributed xgboost
        """
        params = self._gen_xgb_params_dict()
        fit_params = self._gen_fit_params_dict()
        verbose_eval = fit_params.pop('verbose', None)
        params.update(fit_params)
        params['verbose_eval'] = verbose_eval
        classification = self._xgb_cls() == XGBClassifier
        if classification:
            num_classes = int(dataset.select(countDistinct(alias.label)).collect()[0][0])
            if num_classes <= 2:
                params['objective'] = 'binary:logistic'
            else:
                params['objective'] = 'multi:softprob'
                params['num_class'] = num_classes
        else:
            params['objective'] = self.getOrDefault('objective')
        params['num_boost_round'] = self.getOrDefault('n_estimators')
        return params

    @classmethod
    def _get_xgb_train_call_args(cls, train_params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        xgb_train_default_args = _get_default_params_from_func(xgboost.train, _unsupported_train_params)
        booster_params, kwargs_params = ({}, {})
        for key, value in train_params.items():
            if key in xgb_train_default_args:
                kwargs_params[key] = value
            else:
                booster_params[key] = value
        booster_params = {k: v for k, v in booster_params.items() if k not in _non_booster_params}
        return (booster_params, kwargs_params)

    def _prepare_input_columns_and_feature_prop(self, dataset: DataFrame) -> Tuple[List[Column], FeatureProp]:
        label_col = col(self.getOrDefault(self.labelCol)).alias(alias.label)
        select_cols = [label_col]
        features_cols_names = None
        enable_sparse_data_optim = self.getOrDefault(self.enable_sparse_data_optim)
        if enable_sparse_data_optim:
            features_col_name = self.getOrDefault(self.featuresCol)
            features_col_datatype = dataset.schema[features_col_name].dataType
            if not isinstance(features_col_datatype, VectorUDT):
                raise ValueError('If enable_sparse_data_optim is True, the feature column values must be `pyspark.ml.linalg.Vector` type.')
            select_cols.extend(_get_unwrapped_vec_cols(col(features_col_name)))
        elif self.getOrDefault(self.features_cols):
            features_cols_names = self.getOrDefault(self.features_cols)
            features_cols = _validate_and_convert_feature_col_as_float_col_list(dataset, features_cols_names)
            select_cols.extend(features_cols)
        else:
            features_array_col = _validate_and_convert_feature_col_as_array_col(dataset, self.getOrDefault(self.featuresCol))
            select_cols.append(features_array_col)
        if self.isDefined(self.weightCol) and self.getOrDefault(self.weightCol) != '':
            select_cols.append(col(self.getOrDefault(self.weightCol)).alias(alias.weight))
        has_validation_col = False
        if self.isDefined(self.validationIndicatorCol) and self.getOrDefault(self.validationIndicatorCol) != '':
            select_cols.append(col(self.getOrDefault(self.validationIndicatorCol)).alias(alias.valid))
            has_validation_col = True
        if self.isDefined(self.base_margin_col) and self.getOrDefault(self.base_margin_col) != '':
            select_cols.append(col(self.getOrDefault(self.base_margin_col)).alias(alias.margin))
        if self.isDefined(self.qid_col) and self.getOrDefault(self.qid_col) != '':
            select_cols.append(col(self.getOrDefault(self.qid_col)).alias(alias.qid))
        feature_prop = FeatureProp(enable_sparse_data_optim, has_validation_col, features_cols_names)
        return (select_cols, feature_prop)

    def _prepare_input(self, dataset: DataFrame) -> Tuple[DataFrame, FeatureProp]:
        """Prepare the input including column pruning, repartition and so on"""
        select_cols, feature_prop = self._prepare_input_columns_and_feature_prop(dataset)
        dataset = dataset.select(*select_cols)
        num_workers = self.getOrDefault(self.num_workers)
        sc = _get_spark_session().sparkContext
        max_concurrent_tasks = _get_max_num_concurrent_tasks(sc)
        if num_workers > max_concurrent_tasks:
            get_logger(self.__class__.__name__).warning('The num_workers %s set for xgboost distributed training is greater than current max number of concurrent spark task slots, you need wait until more task slots available or you need increase spark cluster workers.', num_workers)
        if self._repartition_needed(dataset) or (self.isDefined(self.validationIndicatorCol) and self.getOrDefault(self.validationIndicatorCol) != ''):
            if self.getOrDefault(self.repartition_random_shuffle):
                dataset = dataset.repartition(num_workers, rand(1))
            else:
                dataset = dataset.repartition(num_workers)
        if self.isDefined(self.qid_col) and self.getOrDefault(self.qid_col) != '':
            dataset = dataset.sortWithinPartitions(alias.qid, ascending=True)
        return (dataset, feature_prop)

    def _get_xgb_parameters(self, dataset: DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        train_params = self._get_distributed_train_params(dataset)
        booster_params, train_call_kwargs_params = self._get_xgb_train_call_args(train_params)
        cpu_per_task = int(_get_spark_session().sparkContext.getConf().get('spark.task.cpus', '1'))
        dmatrix_kwargs = {'nthread': cpu_per_task, 'feature_types': self.getOrDefault('feature_types'), 'feature_names': self.getOrDefault('feature_names'), 'feature_weights': self.getOrDefault('feature_weights'), 'missing': float(self.getOrDefault('missing'))}
        if dmatrix_kwargs['feature_types'] is not None:
            dmatrix_kwargs['enable_categorical'] = True
        booster_params['nthread'] = cpu_per_task
        booster_params = {k: v for k, v in booster_params.items() if v is not None}
        train_call_kwargs_params = {k: v for k, v in train_call_kwargs_params.items() if v is not None}
        dmatrix_kwargs = {k: v for k, v in dmatrix_kwargs.items() if v is not None}
        return (booster_params, train_call_kwargs_params, dmatrix_kwargs)

    def _skip_stage_level_scheduling(self) -> bool:
        """Check if stage-level scheduling is not needed,
        return true to skip stage-level scheduling"""
        if use_cuda(self.getOrDefault(self.device)) or self.getOrDefault(self.use_gpu):
            ss = _get_spark_session()
            sc = ss.sparkContext
            if ss.version < '3.4.0':
                self.logger.info('Stage-level scheduling in xgboost requires spark version 3.4.0+')
                return True
            if not _is_standalone_or_localcluster(sc):
                self.logger.info('Stage-level scheduling in xgboost requires spark standalone or local-cluster mode')
                return True
            executor_cores = sc.getConf().get('spark.executor.cores')
            executor_gpus = sc.getConf().get('spark.executor.resource.gpu.amount')
            if executor_cores is None or executor_gpus is None:
                self.logger.info('Stage-level scheduling in xgboost requires spark.executor.cores, spark.executor.resource.gpu.amount to be set.')
                return True
            if int(executor_cores) == 1:
                self.logger.info('Stage-level scheduling in xgboost requires spark.executor.cores > 1 ')
                return True
            if int(executor_gpus) > 1:
                self.logger.info('Stage-level scheduling in xgboost will not work when spark.executor.resource.gpu.amount>1')
                return True
            task_gpu_amount = sc.getConf().get('spark.task.resource.gpu.amount')
            if task_gpu_amount is None:
                return False
            if float(task_gpu_amount) == float(executor_gpus):
                return True
            return False
        return True

    def _try_stage_level_scheduling(self, rdd: RDD) -> RDD:
        """Try to enable stage-level scheduling"""
        if self._skip_stage_level_scheduling():
            return rdd
        ss = _get_spark_session()
        executor_cores = ss.sparkContext.getConf().get('spark.executor.cores')
        assert executor_cores is not None
        spark_plugins = ss.conf.get('spark.plugins', ' ')
        assert spark_plugins is not None
        spark_rapids_sql_enabled = ss.conf.get('spark.rapids.sql.enabled', 'true')
        assert spark_rapids_sql_enabled is not None
        task_cores = int(executor_cores) if 'com.nvidia.spark.SQLPlugin' in spark_plugins and 'true' == spark_rapids_sql_enabled.lower() else int(executor_cores) // 2 + 1
        task_gpus = 1.0
        treqs = TaskResourceRequests().cpus(task_cores).resource('gpu', task_gpus)
        rp = ResourceProfileBuilder().require(treqs).build
        self.logger.info('XGBoost training tasks require the resource(cores=%s, gpu=%s).', task_cores, task_gpus)
        return rdd.withResources(rp)

    def _fit(self, dataset: DataFrame) -> '_SparkXGBModel':
        self._validate_params()
        dataset, feature_prop = self._prepare_input(dataset)
        booster_params, train_call_kwargs_params, dmatrix_kwargs = self._get_xgb_parameters(dataset)
        run_on_gpu = use_cuda(self.getOrDefault(self.device)) or self.getOrDefault(self.use_gpu)
        is_local = _is_local(_get_spark_session().sparkContext)
        num_workers = self.getOrDefault(self.num_workers)

        def _train_booster(pandas_df_iter: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
            """Takes in an RDD partition and outputs a booster for that partition after
            going through the Rabit Ring protocol

            """
            from pyspark import BarrierTaskContext
            context = BarrierTaskContext.get()
            dev_ordinal = None
            use_qdm = _can_use_qdm(booster_params.get('tree_method', None))
            if run_on_gpu:
                dev_ordinal = context.partitionId() if is_local else _get_gpu_id(context)
                booster_params['device'] = 'cuda:' + str(dev_ordinal)
                use_qdm = use_qdm and is_cudf_available()
                get_logger('XGBoost-PySpark').info('Leveraging %s to train with QDM: %s', booster_params['device'], 'on' if use_qdm else 'off')
            if use_qdm and booster_params.get('max_bin', None) is not None:
                dmatrix_kwargs['max_bin'] = booster_params['max_bin']
            _rabit_args = {}
            if context.partitionId() == 0:
                _rabit_args = _get_rabit_args(context, num_workers)
            worker_message = {'rabit_msg': _rabit_args, 'use_qdm': use_qdm}
            messages = context.allGather(message=json.dumps(worker_message))
            if len(set((json.loads(x)['use_qdm'] for x in messages))) != 1:
                raise RuntimeError("The workers' cudf environments are in-consistent ")
            _rabit_args = json.loads(messages[0])['rabit_msg']
            evals_result: Dict[str, Any] = {}
            with CommunicatorContext(context, **_rabit_args):
                dtrain, dvalid = create_dmatrix_from_partitions(pandas_df_iter, feature_prop.features_cols_names, dev_ordinal, use_qdm, dmatrix_kwargs, enable_sparse_data_optim=feature_prop.enable_sparse_data_optim, has_validation_col=feature_prop.has_validation_col)
                if dvalid is not None:
                    dval = [(dtrain, 'training'), (dvalid, 'validation')]
                else:
                    dval = None
                booster = worker_train(params=booster_params, dtrain=dtrain, evals=dval, evals_result=evals_result, **train_call_kwargs_params)
            context.barrier()
            if context.partitionId() == 0:
                yield pd.DataFrame(data={'config': [booster.save_config()], 'booster': [booster.save_raw('json').decode('utf-8')]})

        def _run_job() -> Tuple[str, str]:
            rdd = dataset.mapInPandas(_train_booster, schema='config string, booster string').rdd.barrier().mapPartitions(lambda x: x)
            rdd_with_resource = self._try_stage_level_scheduling(rdd)
            ret = rdd_with_resource.collect()[0]
            return (ret[0], ret[1])
        get_logger('XGBoost-PySpark').info('Running xgboost-%s on %s workers with\n\tbooster params: %s\n\ttrain_call_kwargs_params: %s\n\tdmatrix_kwargs: %s', xgboost._py_version(), num_workers, booster_params, train_call_kwargs_params, dmatrix_kwargs)
        config, booster = _run_job()
        get_logger('XGBoost-PySpark').info('Finished xgboost training!')
        result_xgb_model = self._convert_to_sklearn_model(bytearray(booster, 'utf-8'), config)
        spark_model = self._create_pyspark_model(result_xgb_model)
        spark_model._resetUid(self.uid)
        return self._copyValues(spark_model)

    def write(self) -> 'SparkXGBWriter':
        """
        Return the writer for saving the estimator.
        """
        return SparkXGBWriter(self)

    @classmethod
    def read(cls) -> 'SparkXGBReader':
        """
        Return the reader for loading the estimator.
        """
        return SparkXGBReader(cls)