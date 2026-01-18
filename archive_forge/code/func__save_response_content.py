from .. import utils
from .._lazyload import requests
import os
import tempfile
import urllib.request
import zipfile
def _save_response_content(response, destination):
    global _CHUNK_SIZE
    if isinstance(destination, str):
        with open(destination, 'wb') as handle:
            _save_response_content(response, handle)
    else:
        for chunk in response.iter_content(_CHUNK_SIZE):
            if chunk:
                destination.write(chunk)