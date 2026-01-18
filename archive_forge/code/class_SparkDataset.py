import json
import logging
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from packaging.version import Version
from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.delta_dataset_source import DeltaDatasetSource
from mlflow.data.digest_utils import get_normalized_md5_digest
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin, PyFuncInputsOutputs
from mlflow.data.spark_dataset_source import SparkDatasetSource
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.base import EvaluationDataset
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.types import Schema
from mlflow.types.utils import _infer_schema
class SparkDataset(Dataset, PyFuncConvertibleDatasetMixin):
    """
    Represents a Spark dataset (e.g. data derived from a Spark Table / file directory or Delta
    Table) for use with MLflow Tracking.
    """

    def __init__(self, df: 'pyspark.sql.DataFrame', source: DatasetSource, targets: Optional[str]=None, name: Optional[str]=None, digest: Optional[str]=None):
        if targets is not None and targets not in df.columns:
            raise MlflowException(f"The specified Spark dataset does not contain the specified targets column '{targets}'.", INVALID_PARAMETER_VALUE)
        self._df = df
        self._targets = targets
        super().__init__(source=source, name=name, digest=digest)

    def _compute_digest(self) -> str:
        """
        Computes a digest for the dataset. Called if the user doesn't supply
        a digest when constructing the dataset.
        """
        import numpy as np
        import pyspark
        if Version(pyspark.__version__) >= Version('3.1.0'):
            semantic_hash = self._df.semanticHash()
        else:
            semantic_hash = self._df._jdf.queryExecution().analyzed().semanticHash()
        return get_normalized_md5_digest([np.int64(semantic_hash)])

    def to_dict(self) -> Dict[str, str]:
        """Create config dictionary for the dataset.

        Returns a string dictionary containing the following fields: name, digest, source, source
        type, schema, and profile.
        """
        schema = json.dumps({'mlflow_colspec': self.schema.to_dict()}) if self.schema else None
        config = super().to_dict()
        config.update({'schema': schema, 'profile': json.dumps(self.profile)})
        return config

    @property
    def df(self):
        """The Spark DataFrame instance.

        Returns:
            The Spark DataFrame instance.

        """
        return self._df

    @property
    def targets(self) -> Optional[str]:
        """The name of the Spark DataFrame column containing targets (labels) for supervised
        learning.

        Returns:
            The string name of the Spark DataFrame column containing targets.
        """
        return self._targets

    @property
    def source(self) -> Union[SparkDatasetSource, DeltaDatasetSource]:
        """
        Spark dataset source information.

        Returns:
            An instance of
            :py:class:`SparkDatasetSource <mlflow.data.spark_dataset_source.SparkDatasetSource>` or
            :py:class:`DeltaDatasetSource <mlflow.data.delta_dataset_source.DeltaDatasetSource>`.
        """
        return self._source

    @property
    def profile(self) -> Optional[Any]:
        """
        A profile of the dataset. May be None if no profile is available.
        """
        try:
            from pyspark.rdd import BoundedFloat
            py_rdd = self.df.rdd
            drdd = py_rdd.mapPartitions(lambda it: [float(sum((1 for i in it)))])
            jrdd = drdd.mapPartitions(lambda it: [float(sum(it))])._to_java_object_rdd()
            jdrdd = drdd.ctx._jvm.JavaDoubleRDD.fromRDD(jrdd.rdd())
            timeout_millis = 5000
            confidence = 0.9
            approx_count_operation = jdrdd.sumApprox(timeout_millis, confidence)
            approx_count_result = approx_count_operation.initialValue()
            approx_count_float = BoundedFloat(mean=approx_count_result.mean(), confidence=approx_count_result.confidence(), low=approx_count_result.low(), high=approx_count_result.high())
            approx_count = int(approx_count_float)
            if approx_count <= 0:
                approx_count = 'unknown'
            return {'approx_count': approx_count}
        except Exception as e:
            _logger.warning('Encountered an unexpected exception while computing Spark dataset profile. Exception: %s', e)

    @cached_property
    def schema(self) -> Optional[Schema]:
        """
        The MLflow ColSpec schema of the Spark dataset.
        """
        try:
            return _infer_schema(self._df)
        except Exception as e:
            _logger.warning('Failed to infer schema for Spark dataset. Exception: %s', e)
            return None

    def to_pyfunc(self) -> PyFuncInputsOutputs:
        """
        Converts the Spark DataFrame to pandas and splits the resulting
        :py:class:`pandas.DataFrame` into: 1. a :py:class:`pandas.DataFrame` of features and
        2. a :py:class:`pandas.Series` of targets.

        To avoid overuse of driver memory, only the first 10,000 DataFrame rows are selected.
        """
        df = self._df.limit(10000).toPandas()
        if self._targets is not None:
            if self._targets not in df.columns:
                raise MlflowException(f"Failed to convert Spark dataset to pyfunc inputs and outputs because the pandas representation of the Spark dataset does not contain the specified targets column '{self._targets}'.", INTERNAL_ERROR)
            inputs = df.drop(columns=self._targets)
            outputs = df[self._targets]
            return PyFuncInputsOutputs(inputs=inputs, outputs=outputs)
        else:
            return PyFuncInputsOutputs(inputs=df, outputs=None)

    def to_evaluation_dataset(self, path=None, feature_names=None) -> EvaluationDataset:
        """
        Converts the dataset to an EvaluationDataset for model evaluation. Required
        for use with mlflow.evaluate().
        """
        return EvaluationDataset(data=self._df.limit(10000).toPandas(), targets=self._targets, path=path, feature_names=feature_names)