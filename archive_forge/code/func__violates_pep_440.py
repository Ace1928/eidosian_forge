import importlib
import re
from packaging.version import InvalidVersion, Version
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.utils.databricks_utils import is_in_databricks_runtime
def _violates_pep_440(ver):
    try:
        Version(ver)
        return False
    except InvalidVersion:
        return True