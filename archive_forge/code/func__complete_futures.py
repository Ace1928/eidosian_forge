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
def _complete_futures(futures_dict, file):
    """
    Waits for the completion of all the futures in the given dictionary and returns
    a tuple of two dictionaries. The first dictionary contains the results of the
    futures (unordered) and the second contains the errors (unordered) that occurred
    during the execution of the futures.
    """
    results = {}
    errors = {}
    with ArtifactProgressBar.chunks(os.path.getsize(file), f'Uploading {file}', MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get()) as pbar:
        for future in as_completed(futures_dict):
            key = futures_dict[future]
            try:
                results[key] = future.result()
                pbar.update()
            except Exception as e:
                errors[key] = repr(e)
    return (results, errors)