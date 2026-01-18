import logging
import os
import secrets
import shutil
import tempfile
import uuid
from contextlib import suppress
from urllib.parse import quote
import requests
from ..spec import AbstractBufferedFile, AbstractFileSystem
from ..utils import infer_storage_options, tokenize
@staticmethod
def _process_info(info):
    info['type'] = info['type'].lower()
    info['size'] = info['length']
    return info