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
class UnstructuredLakeFSLoader(UnstructuredBaseLoader):
    """Load from `lakeFS` as unstructured data."""

    def __init__(self, url: str, repo: str, ref: str='main', path: str='', presign: bool=True, **unstructured_kwargs: Any):
        """Initialize UnstructuredLakeFSLoader.

        Args:

        :param lakefs_access_key:
        :param lakefs_secret_key:
        :param lakefs_endpoint:
        :param repo:
        :param ref:
        """
        super().__init__(**unstructured_kwargs)
        self.url = url
        self.repo = repo
        self.ref = ref
        self.path = path
        self.presign = presign

    def _get_metadata(self) -> dict:
        return {'repo': self.repo, 'ref': self.ref, 'path': self.path}

    def _get_elements(self) -> List:
        from unstructured.partition.auto import partition
        local_prefix = 'local://'
        if self.presign:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = f'{temp_dir}/{self.path.split('/')[-1]}'
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                response = requests.get(self.url)
                response.raise_for_status()
                with open(file_path, mode='wb') as file:
                    file.write(response.content)
                return partition(filename=file_path)
        elif not self.url.startswith(local_prefix):
            raise ValueError("Non pre-signed URLs are supported only with 'local' blockstore")
        else:
            local_path = self.url[len(local_prefix):]
            return partition(filename=local_path)