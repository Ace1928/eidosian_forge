import os
import tempfile
import urllib.parse
from typing import Any, List, Optional
from urllib.parse import urljoin
import requests
from langchain_core.documents import Document
from requests.auth import HTTPBasicAuth
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.unstructured import UnstructuredBaseLoader
def __validate_instance(self) -> None:
    if self.repo is None or self.repo == '':
        raise ValueError('no repository was provided. use `set_repo` to specify a repository')
    if self.ref is None or self.ref == '':
        raise ValueError('no ref was provided. use `set_ref` to specify a ref')
    if self.path is None:
        raise ValueError('no path was provided. use `set_path` to specify a path')