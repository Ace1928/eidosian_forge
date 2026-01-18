import errno
import os
import shutil
import tempfile
from typing import Optional
from absl import logging
from pyglib import stringutil
Saves a discovery document to the on-disc cache with key `api` and `version`.

  Args:
    cache_root: [str], a directory where all cache files are stored.
    api_name: [str], Name of api `discovery_document` to be saved.
    api_version: [str], Version of `discovery_document`.
    discovery_document: [str]. Discovery document as a json string.

  Raises:
    OSError: If an error occurs when the file is written.
  