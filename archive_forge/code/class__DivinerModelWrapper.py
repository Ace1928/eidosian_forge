import logging
import os
import pathlib
import shutil
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import yaml
import mlflow
from mlflow import pyfunc
from mlflow.environment_variables import MLFLOW_DFS_TMP
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import (
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.uri import dbfs_hdfs_uri_to_fuse_path, generate_tmp_dfs_path
class _DivinerModelWrapper:

    def __init__(self, diviner_model):
        self.diviner_model = diviner_model

    def predict(self, dataframe, params: Optional[Dict[str, Any]]=None) -> pd.DataFrame:
        """A method that allows a pyfunc implementation of this flavor to generate forecasted values
        from the end of a trained Diviner model's training series per group.

        The implementation here encapsulates a config-based switch of method calling. In short:
          *  If the ``DataFrame`` supplied to this method contains a column ``groups`` whose
             first row of data is of type List[tuple[str]] (containing the series-identifying
             group keys that were generated to identify a single underlying model during training),
             the caller will resolve to the method ``predict_groups()`` in each of the underlying
             wrapped libraries (i.e., ``GroupedProphet.predict_groups()``).
          *  If the ``DataFrame`` supplied does not contain the column name ``groups``, then the
             specific forecasting method that is primitive-driven (for ``GroupedProphet``, the
             ``predict()`` method mirrors that of ``Prophet``'s, requiring a ``DataFrame``
             submitted with explicit datetime values per group which is not a tenable
             implementation for pyfunc or RESTful serving) is utilized. For ``GroupedProphet``,
             this is the ``.forecast()`` method, while for ``GroupedPmdarima``, this is the
             ``.predict()`` method.

        Args:
            dataframe: A ``pandas.DataFrame`` that contains the required configuration for the
                appropriate ``Diviner`` type.

                For example, for ``GroupedProphet.forecast()``:

                - horizon : int
                - frequency: str

                predict_conf = pd.DataFrame({"horizon": 30, "frequency": "D"}, index=[0])
                forecast = pyfunc.load_pyfunc(model_uri=model_path).predict(predict_conf)

                Will generate 30 days of forecasted values for each group that the model
                was trained on.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                    release without warning.

        Returns:
            A Pandas DataFrame containing the forecasted values for each group key that was
            either trained or declared as a subset with a ``groups`` entry in the ``dataframe``
            configuration argument.
        """
        from diviner import GroupedPmdarima, GroupedProphet
        schema = dataframe.columns.values.tolist()
        conf = dataframe.to_dict(orient='index').get(0)
        horizon = conf.get('horizon', None)
        n_periods = conf.get('n_periods', None)
        if n_periods and horizon and (n_periods != horizon):
            raise MlflowException('The provided prediction configuration contains both `n_periods` and `horizon` with different values. Please provide only one of these integer values.', error_code=INVALID_PARAMETER_VALUE)
        elif not n_periods and horizon:
            n_periods = horizon
        if not n_periods:
            raise MlflowException(f'The provided prediction configuration Pandas DataFrame does not contain either the `n_periods` or `horizon` columns. At least one of these must be specified with a valid integer value. Configuration schema: {schema}.', error_code=INVALID_PARAMETER_VALUE)
        if not isinstance(n_periods, int):
            raise MlflowException(f'The `n_periods` column contains invalid data. Supplied type must be an integer. Type supplied: {type(n_periods)}', error_code=INVALID_PARAMETER_VALUE)
        frequency = conf.get('frequency', None)
        if isinstance(self.diviner_model, GroupedProphet) and (not frequency):
            raise MlflowException(f"Diviner's GroupedProphet model requires a `frequency` value to be submitted in Pandas date_range format. The submitted configuration Pandas DataFrame does not contain a `frequency` column. Configuration schema: {schema}.", error_code=INVALID_PARAMETER_VALUE)
        predict_col = conf.get('predict_col', None)
        predict_groups = conf.get('groups', None)
        if predict_groups and (not isinstance(predict_groups, List)):
            raise MlflowException(f'Specifying a group subset for prediction requires groups to be defined as a [List[(Tuple|List)[<group_keys>]]. Submitted group type: {type(predict_groups)}.', error_code=INVALID_PARAMETER_VALUE)
        if predict_groups and (not isinstance(predict_groups[0], Tuple)):
            predict_groups = [tuple(group) for group in predict_groups]
        if isinstance(self.diviner_model, GroupedProphet):
            if not predict_groups:
                prediction_df = self.diviner_model.forecast(horizon=n_periods, frequency=frequency)
            else:
                group_kwargs = {k: v for k, v in conf.items() if k in {'predict_col', 'on_error'}}
                prediction_df = self.diviner_model.predict_groups(groups=predict_groups, horizon=n_periods, frequency=frequency, **group_kwargs)
            if predict_col is not None:
                prediction_df.rename(columns={'yhat': predict_col}, inplace=True)
        elif isinstance(self.diviner_model, GroupedPmdarima):
            restricted_keys = {'n_periods', 'horizon', 'frequency', 'groups'}
            predict_conf = {k: v for k, v in conf.items() if k not in restricted_keys}
            if not predict_groups:
                prediction_df = self.diviner_model.predict(n_periods=n_periods, **predict_conf)
            else:
                prediction_df = self.diviner_model.predict_groups(groups=predict_groups, n_periods=n_periods, **predict_conf)
        else:
            raise MlflowException(f"The Diviner model instance type '{type(self.diviner_model)}' is not supported in version {mlflow.__version__} of MLflow.", error_code=INVALID_PARAMETER_VALUE)
        return prediction_df