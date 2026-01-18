import os
import hmac
import base64
import hashlib
import binascii
from datetime import datetime, timedelta
from libcloud.utils.py3 import ET, b, httplib, tostring, urlquote, urlencode
from libcloud.utils.xml import fixxpath
from libcloud.utils.files import read_in_chunks
from libcloud.common.azure import AzureConnection, AzureActiveDirectoryConnection
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
def _commit_blocks(self, object_path, chunks, lease, headers, meta_data, content_type, data_hash, object_name, file_path):
    """
        Makes a final commit of the data.
        """
    root = ET.Element('BlockList')
    for block_id in chunks:
        part = ET.SubElement(root, 'Uncommitted')
        part.text = str(block_id)
    data = tostring(root)
    params = {'comp': 'blocklist'}
    headers = headers or {}
    lease.update_headers(headers)
    lease.renew()
    headers['x-ms-blob-content-type'] = self._determine_content_type(content_type, object_name, file_path)
    if data_hash is not None:
        headers['x-ms-blob-content-md5'] = data_hash
    self._update_metadata(headers, meta_data)
    data_hash = self._get_hash_function()
    data_hash.update(data.encode('utf-8'))
    data_hash = base64.b64encode(b(data_hash.digest()))
    headers['Content-MD5'] = data_hash.decode('utf-8')
    headers['Content-Length'] = len(data)
    headers = self._fix_headers(headers)
    response = self.connection.request(object_path, data=data, params=params, headers=headers, method='PUT')
    if response.status != httplib.CREATED:
        raise LibcloudError('Error in blocklist commit', driver=self)
    return response