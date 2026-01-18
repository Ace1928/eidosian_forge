import logging
import math
import os
import posixpath
from abc import abstractmethod
from collections import namedtuple
from concurrent.futures import as_completed
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils import chunk_list
from mlflow.utils.file_utils import (
from mlflow.utils.uri import is_fuse_or_uc_volumes_uri
@abstractmethod
def _get_write_credential_infos(self, remote_file_paths):
    """
        Retrieve write credentials for a batch of remote file paths, including presigned URLs.

        Args:
            remote_file_paths: List of file paths in the remote artifact repository.

        Returns:
            List of ArtifactCredentialInfo objects corresponding to each file path.
        """
    pass