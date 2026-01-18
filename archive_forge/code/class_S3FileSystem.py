import dataclasses
import glob as py_glob
import io
import os
import os.path
import sys
import tempfile
from tensorboard.compat.tensorflow_stub import compat, errors
class S3FileSystem:
    """Provides filesystem access to S3."""

    def __init__(self):
        if not boto3:
            raise ImportError('boto3 must be installed for S3 support.')
        self._s3_endpoint = os.environ.get('S3_ENDPOINT', None)

    def bucket_and_path(self, url):
        """Split an S3-prefixed URL into bucket and path."""
        url = compat.as_str_any(url)
        if url.startswith('s3://'):
            url = url[len('s3://'):]
        idx = url.index('/')
        bucket = url[:idx]
        path = url[idx + 1:]
        return (bucket, path)

    def exists(self, filename):
        """Determines whether a path exists or not."""
        client = boto3.client('s3', endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(filename)
        r = client.list_objects(Bucket=bucket, Prefix=path, Delimiter='/')
        if r.get('Contents') or r.get('CommonPrefixes'):
            return True
        return False

    def join(self, path, *paths):
        """Join paths with a slash."""
        return '/'.join((path,) + paths)

    def read(self, filename, binary_mode=False, size=None, continue_from=None):
        """Reads contents of a file to a string.

        Args:
            filename: string, a path
            binary_mode: bool, read as binary if True, otherwise text
            size: int, number of bytes or characters to read, otherwise
                read all the contents of the file (from the continuation
                marker, if present).
            continue_from: An opaque value returned from a prior invocation of
                `read(...)` marking the last read position, so that reading
                may continue from there.  Otherwise read from the beginning.

        Returns:
            A tuple of `(data, continuation_token)` where `data' provides either
            bytes read from the file (if `binary_mode == true`) or the decoded
            string representation thereof (otherwise), and `continuation_token`
            is an opaque value that can be passed to the next invocation of
            `read(...) ' in order to continue from the last read position.
        """
        s3 = boto3.resource('s3', endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(filename)
        args = {}
        offset = 0
        if continue_from is not None:
            offset = continue_from.get('byte_offset', 0)
        endpoint = ''
        if size is not None:
            endpoint = offset + size
        if offset != 0 or endpoint != '':
            args['Range'] = 'bytes={}-{}'.format(offset, endpoint)
        try:
            stream = s3.Object(bucket, path).get(**args)['Body'].read()
        except botocore.exceptions.ClientError as exc:
            if exc.response['Error']['Code'] in ['416', 'InvalidRange']:
                if size is not None:
                    client = boto3.client('s3', endpoint_url=self._s3_endpoint)
                    obj = client.head_object(Bucket=bucket, Key=path)
                    content_length = obj['ContentLength']
                    endpoint = min(content_length, offset + size)
                if offset == endpoint:
                    stream = b''
                else:
                    args['Range'] = 'bytes={}-{}'.format(offset, endpoint)
                    stream = s3.Object(bucket, path).get(**args)['Body'].read()
            else:
                raise
        continuation_token = {'byte_offset': offset + len(stream)}
        if binary_mode:
            return (bytes(stream), continuation_token)
        else:
            return (stream.decode('utf-8'), continuation_token)

    def write(self, filename, file_content, binary_mode=False):
        """Writes string file contents to a file.

        Args:
            filename: string, a path
            file_content: string, the contents
            binary_mode: bool, write as binary if True, otherwise text
        """
        client = boto3.client('s3', endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(filename)
        if binary_mode:
            if not isinstance(file_content, bytes):
                raise TypeError('File content type must be bytes')
        else:
            file_content = compat.as_bytes(file_content)
        client.put_object(Body=file_content, Bucket=bucket, Key=path)

    def glob(self, filename):
        """Returns a list of files that match the given pattern(s)."""
        star_i = filename.find('*')
        quest_i = filename.find('?')
        if quest_i >= 0:
            raise NotImplementedError('{} not supported by compat glob'.format(filename))
        if star_i != len(filename) - 1:
            return []
        filename = filename[:-1]
        client = boto3.client('s3', endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(filename)
        p = client.get_paginator('list_objects')
        keys = []
        for r in p.paginate(Bucket=bucket, Prefix=path):
            for o in r.get('Contents', []):
                key = o['Key'][len(path):]
                if key:
                    keys.append(filename + key)
        return keys

    def isdir(self, dirname):
        """Returns whether the path is a directory or not."""
        client = boto3.client('s3', endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(dirname)
        if not path.endswith('/'):
            path += '/'
        r = client.list_objects(Bucket=bucket, Prefix=path, Delimiter='/')
        if r.get('Contents') or r.get('CommonPrefixes'):
            return True
        return False

    def listdir(self, dirname):
        """Returns a list of entries contained within a directory."""
        client = boto3.client('s3', endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(dirname)
        p = client.get_paginator('list_objects')
        if not path.endswith('/'):
            path += '/'
        keys = []
        for r in p.paginate(Bucket=bucket, Prefix=path, Delimiter='/'):
            keys.extend((o['Prefix'][len(path):-1] for o in r.get('CommonPrefixes', [])))
            for o in r.get('Contents', []):
                key = o['Key'][len(path):]
                if key:
                    keys.append(key)
        return keys

    def makedirs(self, dirname):
        """Creates a directory and all parent/intermediate directories."""
        if not self.exists(dirname):
            client = boto3.client('s3', endpoint_url=self._s3_endpoint)
            bucket, path = self.bucket_and_path(dirname)
            if not path.endswith('/'):
                path += '/'
            client.put_object(Body='', Bucket=bucket, Key=path)

    def stat(self, filename):
        """Returns file statistics for a given path."""
        client = boto3.client('s3', endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(filename)
        try:
            obj = client.head_object(Bucket=bucket, Key=path)
            return StatData(obj['ContentLength'])
        except botocore.exceptions.ClientError as exc:
            if exc.response['Error']['Code'] == '404':
                raise errors.NotFoundError(None, None, 'Could not find file')
            else:
                raise