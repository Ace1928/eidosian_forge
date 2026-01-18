import json
import logging
import math
import os
import posixpath
import urllib.parse
from mimetypes import guess_type
from mlflow.entities import FileInfo
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialInfo
from mlflow.store.artifact.cloud_artifact_repo import (
from mlflow.store.artifact.s3_artifact_repo import _get_s3_client
from mlflow.utils.file_utils import read_chunk
from mlflow.utils.request_utils import cloud_storage_http_request
from mlflow.utils.rest_utils import augmented_raise_for_status
def _upload_file(self, s3_client, local_file, bucket, key):
    extra_args = {}
    guessed_type, guessed_encoding = guess_type(local_file)
    if guessed_type is not None:
        extra_args['ContentType'] = guessed_type
    if guessed_encoding is not None:
        extra_args['ContentEncoding'] = guessed_encoding
    environ_extra_args = self.get_s3_file_upload_extra_args()
    if environ_extra_args is not None:
        extra_args.update(environ_extra_args)
    s3_client.upload_file(Filename=local_file, Bucket=bucket, Key=key, ExtraArgs=extra_args)