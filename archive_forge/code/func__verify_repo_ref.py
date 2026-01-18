import hashlib
import json
import os
import sys
import tarfile
from io import BytesIO
from os.path import join
from typing import IO, Any, Dict, List, Optional, Set, Tuple, cast
from urllib.error import HTTPError
from urllib.request import urlopen
import onnx
def _verify_repo_ref(repo: str) -> bool:
    """Verifies whether the given model repo can be trusted.
    A model repo can be trusted if it matches onnx/models:main.
    """
    repo_owner, repo_name, repo_ref = _parse_repo_info(repo)
    return repo_owner == 'onnx' and repo_name == 'models' and (repo_ref == 'main')