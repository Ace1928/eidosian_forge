import os
import pathlib
import posixpath
import re
import urllib.parse
import uuid
from typing import Any, Tuple
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.utils.os import is_windows
from mlflow.utils.validation import _validate_db_type_string
def add_databricks_profile_info_to_artifact_uri(artifact_uri, databricks_profile_uri):
    """
    Throws an exception if ``databricks_profile_uri`` is not valid.
    """
    if not databricks_profile_uri or not is_databricks_uri(databricks_profile_uri):
        return artifact_uri
    artifact_uri_parsed = urllib.parse.urlparse(artifact_uri)
    if artifact_uri_parsed.netloc:
        return artifact_uri
    scheme = artifact_uri_parsed.scheme
    if scheme == 'dbfs' or scheme == 'runs' or scheme == 'models':
        if databricks_profile_uri == 'databricks':
            netloc = 'databricks'
        else:
            profile, key_prefix = get_db_info_from_uri(databricks_profile_uri)
            prefix = ':' + key_prefix if key_prefix else ''
            netloc = profile + prefix + '@databricks'
        new_parsed = artifact_uri_parsed._replace(netloc=netloc)
        return urllib.parse.urlunparse(new_parsed)
    else:
        return artifact_uri