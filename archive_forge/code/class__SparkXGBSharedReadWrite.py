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
class _SparkXGBSharedReadWrite:

    @staticmethod
    def saveMetadata(instance: Union[_SparkXGBEstimator, _SparkXGBModel], path: str, sc: SparkContext, logger: logging.Logger, extraMetadata: Optional[Dict[str, Any]]=None) -> None:
        """
        Save the metadata of an xgboost.spark._SparkXGBEstimator or
        xgboost.spark._SparkXGBModel.
        """
        instance._validate_params()
        skipParams = ['callbacks', 'xgb_model']
        jsonParams = {}
        for p, v in instance._paramMap.items():
            if p.name not in skipParams:
                jsonParams[p.name] = v
        extraMetadata = extraMetadata or {}
        callbacks = instance.getOrDefault('callbacks')
        if callbacks is not None:
            logger.warning('The callbacks parameter is saved using cloudpickle and it is not a fully self-contained format. It may fail to load with different versions of dependencies.')
            serialized_callbacks = base64.encodebytes(cloudpickle.dumps(callbacks)).decode('ascii')
            extraMetadata['serialized_callbacks'] = serialized_callbacks
        init_booster = instance.getOrDefault('xgb_model')
        if init_booster is not None:
            extraMetadata['init_booster'] = _INIT_BOOSTER_SAVE_PATH
        DefaultParamsWriter.saveMetadata(instance, path, sc, extraMetadata=extraMetadata, paramMap=jsonParams)
        if init_booster is not None:
            ser_init_booster = serialize_booster(init_booster)
            save_path = os.path.join(path, _INIT_BOOSTER_SAVE_PATH)
            _get_spark_session().createDataFrame([(ser_init_booster,)], ['init_booster']).write.parquet(save_path)

    @staticmethod
    def loadMetadataAndInstance(pyspark_xgb_cls: Union[Type[_SparkXGBEstimator], Type[_SparkXGBModel]], path: str, sc: SparkContext, logger: logging.Logger) -> Tuple[Dict[str, Any], Union[_SparkXGBEstimator, _SparkXGBModel]]:
        """
        Load the metadata and the instance of an xgboost.spark._SparkXGBEstimator or
        xgboost.spark._SparkXGBModel.

        :return: a tuple of (metadata, instance)
        """
        metadata = DefaultParamsReader.loadMetadata(path, sc, expectedClassName=get_class_name(pyspark_xgb_cls))
        pyspark_xgb = pyspark_xgb_cls()
        DefaultParamsReader.getAndSetParams(pyspark_xgb, metadata)
        if 'serialized_callbacks' in metadata:
            serialized_callbacks = metadata['serialized_callbacks']
            try:
                callbacks = cloudpickle.loads(base64.decodebytes(serialized_callbacks.encode('ascii')))
                pyspark_xgb.set(pyspark_xgb.callbacks, callbacks)
            except Exception as e:
                logger.warning(f'Fails to load the callbacks param due to {e}. Please set the callbacks param manually for the loaded estimator.')
        if 'init_booster' in metadata:
            load_path = os.path.join(path, metadata['init_booster'])
            ser_init_booster = _get_spark_session().read.parquet(load_path).collect()[0].init_booster
            init_booster = deserialize_booster(ser_init_booster)
            pyspark_xgb.set(pyspark_xgb.xgb_model, init_booster)
        pyspark_xgb._resetUid(metadata['uid'])
        return (metadata, pyspark_xgb)