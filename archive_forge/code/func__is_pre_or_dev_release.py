import importlib
import re
from packaging.version import InvalidVersion, Version
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.utils.databricks_utils import is_in_databricks_runtime
def _is_pre_or_dev_release(ver):
    v = Version(ver)
    return v.is_devrelease or v.is_prerelease