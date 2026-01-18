from .. import utils
from .._lazyload import requests
import os
import tempfile
import urllib.request
import zipfile
def _google_drive_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None