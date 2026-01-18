import json
import os
import posixpath
import urllib.parse
from datetime import datetime
from functools import lru_cache
from mimetypes import guess_type
from mlflow.entities import FileInfo
from mlflow.entities.multipart_upload import (
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import (
from mlflow.utils.file_utils import relative_path_to_artifact_path
class S3ArtifactRepository(ArtifactRepository, MultipartUploadMixin):
    """Stores artifacts on Amazon S3."""

    def __init__(self, artifact_uri, access_key_id=None, secret_access_key=None, session_token=None):
        super().__init__(artifact_uri)
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._session_token = session_token

    def _get_s3_client(self):
        return _get_s3_client(access_key_id=self._access_key_id, secret_access_key=self._secret_access_key, session_token=self._session_token)

    def parse_s3_compliant_uri(self, uri):
        """Parse an S3 URI, returning (bucket, path)"""
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != 's3':
            raise Exception(f'Not an S3 URI: {uri}')
        path = parsed.path
        if path.startswith('/'):
            path = path[1:]
        return (parsed.netloc, path)

    @staticmethod
    def get_s3_file_upload_extra_args():
        s3_file_upload_extra_args = MLFLOW_S3_UPLOAD_EXTRA_ARGS.get()
        if s3_file_upload_extra_args:
            return json.loads(s3_file_upload_extra_args)
        else:
            return None

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

    def log_artifact(self, local_file, artifact_path=None):
        bucket, dest_path = self.parse_s3_compliant_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(dest_path, os.path.basename(local_file))
        self._upload_file(s3_client=self._get_s3_client(), local_file=local_file, bucket=bucket, key=dest_path)

    def log_artifacts(self, local_dir, artifact_path=None):
        bucket, dest_path = self.parse_s3_compliant_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        s3_client = self._get_s3_client()
        local_dir = os.path.abspath(local_dir)
        for root, _, filenames in os.walk(local_dir):
            upload_path = dest_path
            if root != local_dir:
                rel_path = os.path.relpath(root, local_dir)
                rel_path = relative_path_to_artifact_path(rel_path)
                upload_path = posixpath.join(dest_path, rel_path)
            if not filenames:
                s3_client.put_object(Bucket=bucket, Key=upload_path + '/')
            for f in filenames:
                self._upload_file(s3_client=s3_client, local_file=os.path.join(root, f), bucket=bucket, key=posixpath.join(upload_path, f))

    def list_artifacts(self, path=None):
        bucket, artifact_path = self.parse_s3_compliant_uri(self.artifact_uri)
        dest_path = artifact_path
        if path:
            dest_path = posixpath.join(dest_path, path)
        infos = []
        prefix = dest_path + '/' if dest_path else ''
        s3_client = self._get_s3_client()
        paginator = s3_client.get_paginator('list_objects_v2')
        results = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/')
        for result in results:
            for obj in result.get('CommonPrefixes', []):
                subdir_path = obj.get('Prefix')
                self._verify_listed_object_contains_artifact_path_prefix(listed_object_path=subdir_path, artifact_path=artifact_path)
                subdir_rel_path = posixpath.relpath(path=subdir_path, start=artifact_path)
                if subdir_rel_path.endswith('/'):
                    subdir_rel_path = subdir_rel_path[:-1]
                infos.append(FileInfo(subdir_rel_path, True, None))
            for obj in result.get('Contents', []):
                file_path = obj.get('Key')
                self._verify_listed_object_contains_artifact_path_prefix(listed_object_path=file_path, artifact_path=artifact_path)
                file_rel_path = posixpath.relpath(path=file_path, start=artifact_path)
                file_size = int(obj.get('Size'))
                infos.append(FileInfo(file_rel_path, False, file_size))
        return sorted(infos, key=lambda f: f.path)

    @staticmethod
    def _verify_listed_object_contains_artifact_path_prefix(listed_object_path, artifact_path):
        if not listed_object_path.startswith(artifact_path):
            raise MlflowException(f'The path of the listed S3 object does not begin with the specified artifact path. Artifact path: {artifact_path}. Object path: {listed_object_path}.')

    def _download_file(self, remote_file_path, local_path):
        bucket, s3_root_path = self.parse_s3_compliant_uri(self.artifact_uri)
        s3_full_path = posixpath.join(s3_root_path, remote_file_path)
        s3_client = self._get_s3_client()
        s3_client.download_file(bucket, s3_full_path, local_path)

    def delete_artifacts(self, artifact_path=None):
        bucket, dest_path = self.parse_s3_compliant_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        s3_client = self._get_s3_client()
        list_objects = s3_client.list_objects(Bucket=bucket, Prefix=dest_path).get('Contents', [])
        for to_delete_obj in list_objects:
            file_path = to_delete_obj.get('Key')
            self._verify_listed_object_contains_artifact_path_prefix(listed_object_path=file_path, artifact_path=dest_path)
            s3_client.delete_object(Bucket=bucket, Key=file_path)

    def create_multipart_upload(self, local_file, num_parts=1, artifact_path=None):
        bucket, dest_path = self.parse_s3_compliant_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(dest_path, os.path.basename(local_file))
        s3_client = self._get_s3_client()
        create_response = s3_client.create_multipart_upload(Bucket=bucket, Key=dest_path)
        upload_id = create_response['UploadId']
        credentials = []
        for i in range(1, num_parts + 1):
            url = s3_client.generate_presigned_url('upload_part', Params={'Bucket': bucket, 'Key': dest_path, 'PartNumber': i, 'UploadId': upload_id})
            credentials.append(MultipartUploadCredential(url=url, part_number=i, headers={}))
        return CreateMultipartUploadResponse(credentials=credentials, upload_id=upload_id)

    def complete_multipart_upload(self, local_file, upload_id, parts=None, artifact_path=None):
        bucket, dest_path = self.parse_s3_compliant_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(dest_path, os.path.basename(local_file))
        parts = [{'PartNumber': part.part_number, 'ETag': part.etag} for part in parts]
        s3_client = self._get_s3_client()
        s3_client.complete_multipart_upload(Bucket=bucket, Key=dest_path, UploadId=upload_id, MultipartUpload={'Parts': parts})

    def abort_multipart_upload(self, local_file, upload_id, artifact_path=None):
        bucket, dest_path = self.parse_s3_compliant_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(dest_path, os.path.basename(local_file))
        s3_client = self._get_s3_client()
        s3_client.abort_multipart_upload(Bucket=bucket, Key=dest_path, UploadId=upload_id)