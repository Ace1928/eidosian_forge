import json
import logging
from functools import cached_property
from typing import Any, Dict, Optional, Union
import pandas as pd
from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.digest_utils import compute_pandas_digest
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin, PyFuncInputsOutputs
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.base import EvaluationDataset
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types import Schema
from mlflow.types.utils import _infer_schema
class PandasDataset(Dataset, PyFuncConvertibleDatasetMixin):
    """
    Represents a Pandas DataFrame for use with MLflow Tracking.
    """

    def __init__(self, df: pd.DataFrame, source: DatasetSource, targets: Optional[str]=None, name: Optional[str]=None, digest: Optional[str]=None, predictions: Optional[str]=None):
        """
        Args:
            df: A pandas DataFrame.
            source: The source of the pandas DataFrame.
            targets: The name of the target column. Optional.
            name: The name of the dataset. E.g. "wiki_train". If unspecified, a name is
                automatically generated.
            digest: The digest (hash, fingerprint) of the dataset. If unspecified, a digest
                is automatically computed.
            predictions: Optional. The name of the column containing model predictions,
                if the dataset contains model predictions. If specified, this column
                must be present in the dataframe (``df``).
        """
        if targets is not None and targets not in df.columns:
            raise MlflowException(f"The specified pandas DataFrame does not contain the specified targets column '{targets}'.", INVALID_PARAMETER_VALUE)
        if predictions is not None and predictions not in df.columns:
            raise MlflowException(f"The specified pandas DataFrame does not contain the specified predictions column '{predictions}'.", INVALID_PARAMETER_VALUE)
        self._df = df
        self._targets = targets
        self._predictions = predictions
        super().__init__(source=source, name=name, digest=digest)

    def _compute_digest(self) -> str:
        """
        Computes a digest for the dataset. Called if the user doesn't supply
        a digest when constructing the dataset.
        """
        return compute_pandas_digest(self._df)

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
    def df(self) -> pd.DataFrame:
        """
        The underlying pandas DataFrame.
        """
        return self._df

    @property
    def source(self) -> DatasetSource:
        """
        The source of the dataset.
        """
        return self._source

    @property
    def targets(self) -> Optional[str]:
        """
        The name of the target column. May be ``None`` if no target column is available.
        """
        return self._targets

    @property
    def predictions(self) -> Optional[str]:
        """
        The name of the predictions column. May be ``None`` if no predictions column is available.
        """
        return self._predictions

    @property
    def profile(self) -> Optional[Any]:
        """
        A profile of the dataset. May be ``None`` if a profile cannot be computed.
        """
        return {'num_rows': len(self._df), 'num_elements': int(self._df.size)}

    @cached_property
    def schema(self) -> Optional[Schema]:
        """
        An instance of :py:class:`mlflow.types.Schema` representing the tabular dataset. May be
        ``None`` if the schema cannot be inferred from the dataset.
        """
        try:
            return _infer_schema(self._df)
        except Exception as e:
            _logger.warning('Failed to infer schema for Pandas dataset. Exception: %s', e)
            return None

    def to_pyfunc(self) -> PyFuncInputsOutputs:
        """
        Converts the dataset to a collection of pyfunc inputs and outputs for model
        evaluation. Required for use with mlflow.evaluate().
        """
        if self._targets:
            inputs = self._df.drop(columns=[self._targets])
            outputs = self._df[self._targets]
            return PyFuncInputsOutputs(inputs, outputs)
        else:
            return PyFuncInputsOutputs(self._df)

    def to_evaluation_dataset(self, path=None, feature_names=None) -> EvaluationDataset:
        """
        Converts the dataset to an EvaluationDataset for model evaluation. Required
        for use with mlflow.evaluate().
        """
        return EvaluationDataset(data=self._df, targets=self._targets, path=path, feature_names=feature_names, predictions=self._predictions)