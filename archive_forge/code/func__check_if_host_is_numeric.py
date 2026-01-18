import re
from urllib.parse import urlparse, urlunparse
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.http_artifact_repo import HttpArtifactRepository
from mlflow.tracking._tracking_service.utils import get_tracking_uri
def _check_if_host_is_numeric(hostname):
    if hostname:
        try:
            float(hostname)
            return True
        except ValueError:
            return False
    else:
        return False