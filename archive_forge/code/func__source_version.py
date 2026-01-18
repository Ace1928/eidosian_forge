import logging
from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.tracking.context.default_context import _get_main_file
from mlflow.utils.git_utils import get_git_commit
from mlflow.utils.mlflow_tags import MLFLOW_GIT_COMMIT
@property
def _source_version(self):
    if 'source_version' not in self._cache:
        self._cache['source_version'] = _get_source_version()
    return self._cache['source_version']