import json
import logging
import os
import re
import tempfile
import time
from abc import ABC
from io import StringIO
from pathlib import Path
from typing import (
from urllib.parse import urlparse
import requests
from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers.pdf import (
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
@staticmethod
def _is_s3_url(url: str) -> bool:
    """check if the url is S3"""
    try:
        result = urlparse(url)
        if result.scheme == 's3' and result.netloc:
            return True
        return False
    except ValueError:
        return False