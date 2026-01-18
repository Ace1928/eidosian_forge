import json
import os
import re
import shutil
import sys
import tempfile
import traceback
import warnings
from concurrent import futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from uuid import uuid4
import huggingface_hub
import requests
from huggingface_hub import (
from huggingface_hub.file_download import REGEX_COMMIT_HASH, http_get
from huggingface_hub.utils import (
from huggingface_hub.utils._deprecation import _deprecate_method
from requests.exceptions import HTTPError
from . import __version__, logging
from .generic import working_or_temp_dir
from .import_utils import (
from .logging import tqdm
def _upload_modified_files(self, working_dir: Union[str, os.PathLike], repo_id: str, files_timestamps: Dict[str, float], commit_message: Optional[str]=None, token: Optional[Union[bool, str]]=None, create_pr: bool=False, revision: str=None, commit_description: str=None):
    """
        Uploads all modified files in `working_dir` to `repo_id`, based on `files_timestamps`.
        """
    if commit_message is None:
        if 'Model' in self.__class__.__name__:
            commit_message = 'Upload model'
        elif 'Config' in self.__class__.__name__:
            commit_message = 'Upload config'
        elif 'Tokenizer' in self.__class__.__name__:
            commit_message = 'Upload tokenizer'
        elif 'FeatureExtractor' in self.__class__.__name__:
            commit_message = 'Upload feature extractor'
        elif 'Processor' in self.__class__.__name__:
            commit_message = 'Upload processor'
        else:
            commit_message = f'Upload {self.__class__.__name__}'
    modified_files = [f for f in os.listdir(working_dir) if f not in files_timestamps or os.path.getmtime(os.path.join(working_dir, f)) > files_timestamps[f]]
    modified_files = [f for f in modified_files if os.path.isfile(os.path.join(working_dir, f)) or os.path.isdir(os.path.join(working_dir, f))]
    operations = []
    for file in modified_files:
        if os.path.isdir(os.path.join(working_dir, file)):
            for f in os.listdir(os.path.join(working_dir, file)):
                operations.append(CommitOperationAdd(path_or_fileobj=os.path.join(working_dir, file, f), path_in_repo=os.path.join(file, f)))
        else:
            operations.append(CommitOperationAdd(path_or_fileobj=os.path.join(working_dir, file), path_in_repo=file))
    if revision is not None:
        create_branch(repo_id=repo_id, branch=revision, token=token, exist_ok=True)
    logger.info(f'Uploading the following files to {repo_id}: {','.join(modified_files)}')
    return create_commit(repo_id=repo_id, operations=operations, commit_message=commit_message, commit_description=commit_description, token=token, create_pr=create_pr, revision=revision)