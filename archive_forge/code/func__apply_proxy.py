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
def _apply_proxy(self, location):
    if self.proxy and callable(self.proxy):
        location = self.proxy(location)
    elif self.proxy:
        for k, v in self.proxy.items():
            location = location.replace(k, v, 1)
    return location