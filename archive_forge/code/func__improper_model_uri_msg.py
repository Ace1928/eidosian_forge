import urllib.parse
from typing import NamedTuple, Optional
import mlflow.tracking
from mlflow.exceptions import MlflowException
from mlflow.utils.uri import get_databricks_profile_uri_from_artifact_uri, is_databricks_uri
def _improper_model_uri_msg(uri):
    return f'Not a proper models:/ URI: {uri}. ' + "Models URIs must be of the form 'models:/model_name/suffix' " + "or 'models:/model_name@alias' where suffix is a model version, stage, " + "or the string '%s' and where alias is a registered model alias. " % _MODELS_URI_SUFFIX_LATEST + 'Only one of suffix or alias can be defined at a time.'