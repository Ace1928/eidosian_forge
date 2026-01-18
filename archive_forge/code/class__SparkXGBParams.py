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
class _SparkXGBParams(HasFeaturesCol, HasLabelCol, HasWeightCol, HasPredictionCol, HasValidationIndicatorCol, HasArbitraryParamsDict, HasBaseMarginCol, HasFeaturesCols, HasEnableSparseDataOptim, HasQueryIdCol, HasContribPredictionCol):
    num_workers = Param(Params._dummy(), 'num_workers', 'The number of XGBoost workers. Each XGBoost worker corresponds to one spark task.', TypeConverters.toInt)
    device = Param(Params._dummy(), 'device', 'The device type for XGBoost executors. Available options are `cpu`,`cuda` and `gpu`. Set `device` to `cuda` or `gpu` if the executors are running on GPU instances. Currently, only one GPU per task is supported.', TypeConverters.toString)
    use_gpu = Param(Params._dummy(), 'use_gpu', 'Deprecated, use `device` instead. A boolean variable. Set use_gpu=true if the executors are running on GPU instances. Currently, only one GPU per task is supported.', TypeConverters.toBoolean)
    force_repartition = Param(Params._dummy(), 'force_repartition', 'A boolean variable. Set force_repartition=true if you ' + 'want to force the input dataset to be repartitioned before XGBoost training.' + 'Note: The auto repartitioning judgement is not fully accurate, so it is recommended' + 'to have force_repartition be True.', TypeConverters.toBoolean)
    repartition_random_shuffle = Param(Params._dummy(), 'repartition_random_shuffle', 'A boolean variable. Set repartition_random_shuffle=true if you want to random shuffle dataset when repartitioning is required. By default is True.', TypeConverters.toBoolean)
    feature_names = Param(Params._dummy(), 'feature_names', 'A list of str to specify feature names.', TypeConverters.toList)

    def set_device(self, value: str) -> '_SparkXGBParams':
        """Set device, optional value: cpu, cuda, gpu"""
        _check_distributed_params({'device': value})
        assert value in ('cpu', 'cuda', 'gpu')
        self.set(self.device, value)
        return self

    @classmethod
    def _xgb_cls(cls) -> Type[XGBModel]:
        """
        Subclasses should override this method and
        returns an xgboost.XGBModel subclass
        """
        raise NotImplementedError()

    @classmethod
    def _get_xgb_params_default(cls) -> Dict[str, Any]:
        """Get the xgboost.sklearn.XGBModel default parameters and filter out some"""
        xgb_model_default = cls._xgb_cls()()
        params_dict = xgb_model_default.get_params()
        filtered_params_dict = {k: params_dict[k] for k in params_dict if k not in _unsupported_xgb_params}
        filtered_params_dict['n_estimators'] = DEFAULT_N_ESTIMATORS
        return filtered_params_dict

    def _set_xgb_params_default(self) -> None:
        """Set xgboost parameters into spark parameters"""
        filtered_params_dict = self._get_xgb_params_default()
        self._setDefault(**filtered_params_dict)

    def _gen_xgb_params_dict(self, gen_xgb_sklearn_estimator_param: bool=False) -> Dict[str, Any]:
        """Generate the xgboost parameters which will be passed into xgboost library"""
        xgb_params = {}
        non_xgb_params = set(_pyspark_specific_params) | self._get_fit_params_default().keys() | self._get_predict_params_default().keys()
        if not gen_xgb_sklearn_estimator_param:
            non_xgb_params |= set(_non_booster_params)
        for param in self.extractParamMap():
            if param.name not in non_xgb_params:
                xgb_params[param.name] = self.getOrDefault(param)
        arbitrary_params_dict = self.getOrDefault(self.getParam('arbitrary_params_dict'))
        xgb_params.update(arbitrary_params_dict)
        return xgb_params

    @classmethod
    def _get_fit_params_default(cls) -> Dict[str, Any]:
        """Get the xgboost.XGBModel().fit() parameters"""
        fit_params = _get_default_params_from_func(cls._xgb_cls().fit, _unsupported_fit_params)
        return fit_params

    def _set_fit_params_default(self) -> None:
        """Get the xgboost.XGBModel().fit() parameters and set them to spark parameters"""
        filtered_params_dict = self._get_fit_params_default()
        self._setDefault(**filtered_params_dict)

    def _gen_fit_params_dict(self) -> Dict[str, Any]:
        """Generate the fit parameters which will be passed into fit function"""
        fit_params_keys = self._get_fit_params_default().keys()
        fit_params = {}
        for param in self.extractParamMap():
            if param.name in fit_params_keys:
                fit_params[param.name] = self.getOrDefault(param)
        return fit_params

    @classmethod
    def _get_predict_params_default(cls) -> Dict[str, Any]:
        """Get the parameters from xgboost.XGBModel().predict()"""
        predict_params = _get_default_params_from_func(cls._xgb_cls().predict, _unsupported_predict_params)
        return predict_params

    def _set_predict_params_default(self) -> None:
        """Get the parameters from xgboost.XGBModel().predict() and
        set them into spark parameters"""
        filtered_params_dict = self._get_predict_params_default()
        self._setDefault(**filtered_params_dict)

    def _gen_predict_params_dict(self) -> Dict[str, Any]:
        """Generate predict parameters which will be passed into xgboost.XGBModel().predict()"""
        predict_params_keys = self._get_predict_params_default().keys()
        predict_params = {}
        for param in self.extractParamMap():
            if param.name in predict_params_keys:
                predict_params[param.name] = self.getOrDefault(param)
        return predict_params

    def _validate_gpu_params(self) -> None:
        """Validate the gpu parameters and gpu configurations"""
        if use_cuda(self.getOrDefault(self.device)) or self.getOrDefault(self.use_gpu):
            ss = _get_spark_session()
            sc = ss.sparkContext
            if _is_local(sc):
                get_logger(self.__class__.__name__).warning('You have enabled GPU in spark local mode. Please make sure your local node has at least %d GPUs', self.getOrDefault(self.num_workers))
            else:
                executor_gpus = sc.getConf().get('spark.executor.resource.gpu.amount')
                if executor_gpus is None:
                    raise ValueError('The `spark.executor.resource.gpu.amount` is required for training on GPU.')
                if not (ss.version >= '3.4.0' and _is_standalone_or_localcluster(sc)):
                    gpu_per_task = sc.getConf().get('spark.task.resource.gpu.amount')
                    if gpu_per_task is not None:
                        if float(gpu_per_task) < 1.0:
                            raise ValueError("XGBoost doesn't support GPU fractional configurations. Please set `spark.task.resource.gpu.amount=spark.executor.resource.gpu.amount`")
                        if float(gpu_per_task) > 1.0:
                            get_logger(self.__class__.__name__).warning('%s GPUs for each Spark task is configured, but each XGBoost training task uses only 1 GPU.', gpu_per_task)
                    else:
                        raise ValueError('The `spark.task.resource.gpu.amount` is required for training on GPU.')

    def _validate_params(self) -> None:
        init_model = self.getOrDefault('xgb_model')
        if init_model is not None and (not isinstance(init_model, Booster)):
            raise ValueError('The xgb_model param must be set with a `xgboost.core.Booster` instance.')
        if self.getOrDefault(self.num_workers) < 1:
            raise ValueError(f'Number of workers was {self.getOrDefault(self.num_workers)}.It cannot be less than 1 [Default is 1]')
        tree_method = self.getOrDefault(self.getParam('tree_method'))
        if tree_method == 'exact':
            raise ValueError('The `exact` tree method is not supported for distributed systems.')
        if self.getOrDefault(self.features_cols):
            if not use_cuda(self.getOrDefault(self.device)) and (not self.getOrDefault(self.use_gpu)):
                raise ValueError('features_col param with list value requires `device=cuda`.')
        if self.getOrDefault('objective') is not None:
            if not isinstance(self.getOrDefault('objective'), str):
                raise ValueError("Only string type 'objective' param is allowed.")
        eval_metric = 'eval_metric'
        if self.getOrDefault(eval_metric) is not None:
            if not (isinstance(self.getOrDefault(eval_metric), str) or (isinstance(self.getOrDefault(eval_metric), List) and all((isinstance(metric, str) for metric in self.getOrDefault(eval_metric))))):
                raise ValueError("Only string type or list of string type 'eval_metric' param is allowed.")
        if self.getOrDefault('early_stopping_rounds') is not None:
            if not (self.isDefined(self.validationIndicatorCol) and self.getOrDefault(self.validationIndicatorCol) != ''):
                raise ValueError("If 'early_stopping_rounds' param is set, you need to set 'validation_indicator_col' param as well.")
        if self.getOrDefault(self.enable_sparse_data_optim):
            if self.getOrDefault('missing') != 0.0:
                raise ValueError('If enable_sparse_data_optim is True, missing param != 0 is not supported.')
            if self.getOrDefault(self.features_cols):
                raise ValueError('If enable_sparse_data_optim is True, you cannot set multiple feature columns but you should set one feature column with values of `pyspark.ml.linalg.Vector` type.')
        self._validate_gpu_params()