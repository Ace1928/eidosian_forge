import logging
import math
import os
import posixpath
from abc import abstractmethod
from collections import namedtuple
from concurrent.futures import as_completed
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils import chunk_list
from mlflow.utils.file_utils import (
from mlflow.utils.uri import is_fuse_or_uc_volumes_uri
def _parallelized_download_from_cloud(self, file_size, remote_file_path, local_path):
    read_credentials = self._get_read_credential_infos([remote_file_path])
    assert len(read_credentials) == 1
    cloud_credential_info = read_credentials[0]
    with remove_on_error(local_path):
        parallel_download_subproc_env = os.environ.copy()
        failed_downloads = parallelized_download_file_using_http_uri(thread_pool_executor=self.chunk_thread_pool, http_uri=cloud_credential_info.signed_uri, download_path=local_path, remote_file_path=remote_file_path, file_size=file_size, uri_type=cloud_credential_info.type, chunk_size=MLFLOW_MULTIPART_DOWNLOAD_CHUNK_SIZE.get(), env=parallel_download_subproc_env, headers=self._extract_headers_from_credentials(cloud_credential_info.headers))
        if failed_downloads:
            new_cloud_creds = self._get_read_credential_infos([remote_file_path])[0]
            new_signed_uri = new_cloud_creds.signed_uri
            new_headers = self._extract_headers_from_credentials(new_cloud_creds.headers)
            download_chunk_retries(chunks=list(failed_downloads), headers=new_headers, http_uri=new_signed_uri, download_path=local_path)