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
class PushInProgress:
    """
    Internal class to keep track of a push in progress (which might contain multiple `Future` jobs).
    """

    def __init__(self, jobs: Optional[futures.Future]=None) -> None:
        self.jobs = [] if jobs is None else jobs

    def is_done(self):
        return all((job.done() for job in self.jobs))

    def wait_until_done(self):
        futures.wait(self.jobs)

    def cancel(self) -> None:
        self.jobs = [job for job in self.jobs if not (job.cancel() or job.done())]