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
class WebHDFile(AbstractBufferedFile):
    """A file living in HDFS over webHDFS"""

    def __init__(self, fs, path, **kwargs):
        super().__init__(fs, path, **kwargs)
        kwargs = kwargs.copy()
        if kwargs.get('permissions', None) is None:
            kwargs.pop('permissions', None)
        if kwargs.get('replication', None) is None:
            kwargs.pop('replication', None)
        self.permissions = kwargs.pop('permissions', 511)
        tempdir = kwargs.pop('tempdir')
        if kwargs.pop('autocommit', False) is False:
            self.target = self.path
            self.path = os.path.join(tempdir, str(uuid.uuid4()))

    def _upload_chunk(self, final=False):
        """Write one part of a multi-block file upload

        Parameters
        ==========
        final: bool
            This is the last block, so should complete file, if
            self.autocommit is True.
        """
        out = self.fs.session.post(self.location, data=self.buffer.getvalue(), headers={'content-type': 'application/octet-stream'})
        out.raise_for_status()
        return True

    def _initiate_upload(self):
        """Create remote file/upload"""
        kwargs = self.kwargs.copy()
        if 'a' in self.mode:
            op, method = ('APPEND', 'POST')
        else:
            op, method = ('CREATE', 'PUT')
            kwargs['overwrite'] = 'true'
        out = self.fs._call(op, method, self.path, redirect=False, **kwargs)
        location = self.fs._apply_proxy(out.headers['Location'])
        if 'w' in self.mode:
            out2 = self.fs.session.put(location, headers={'content-type': 'application/octet-stream'})
            out2.raise_for_status()
            out2 = self.fs._call('APPEND', 'POST', self.path, redirect=False, **kwargs)
            self.location = self.fs._apply_proxy(out2.headers['Location'])

    def _fetch_range(self, start, end):
        start = max(start, 0)
        end = min(self.size, end)
        if start >= end or start >= self.size:
            return b''
        out = self.fs._call('OPEN', path=self.path, offset=start, length=end - start, redirect=False)
        out.raise_for_status()
        if 'Location' in out.headers:
            location = out.headers['Location']
            out2 = self.fs.session.get(self.fs._apply_proxy(location))
            return out2.content
        else:
            return out.content

    def commit(self):
        self.fs.mv(self.path, self.target)

    def discard(self):
        self.fs.rm(self.path)