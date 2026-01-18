import os
import hmac
import time
import base64
from typing import Dict, Optional
from hashlib import sha1
from datetime import datetime
import libcloud.utils.py3
from libcloud.utils.py3 import b, httplib, tostring, urlquote, urlencode
from libcloud.utils.xml import findtext, fixxpath
from libcloud.common.aws import (
from libcloud.common.base import RawResponse, ConnectionUserAndKey
from libcloud.utils.files import read_in_chunks
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
def _commit_multipart(self, container, object_name, upload_id, chunks):
    """
        Makes a final commit of the data.

        :param container: The destination container
        :type container: :class:`Container`

        :param object_name: The name of the object which we are uploading
        :type object_name: ``str``

        :param upload_id: The upload id allocated for this multipart upload
        :type upload_id: ``str``

        :param chunks: A list of (chunk_number, chunk_hash) tuples.
        :type chunks: ``list``

        :return: The server side hash of the uploaded data
        :rtype: ``str``
        """
    root = Element('CompleteMultipartUpload')
    for count, etag in chunks:
        part = SubElement(root, 'Part')
        part_no = SubElement(part, 'PartNumber')
        part_no.text = str(count)
        etag_id = SubElement(part, 'ETag')
        etag_id.text = str(etag)
    data = tostring(root)
    headers = {'Content-Length': len(data)}
    params = {'uploadId': upload_id}
    request_path = self._get_object_path(container, object_name)
    response = self.connection.request(request_path, headers=headers, params=params, data=data, method='POST')
    if response.status != httplib.OK:
        element = response.object
        code, message = response._parse_error_details(element=element)
        msg = 'Error in multipart commit: {} ({})'.format(message, code)
        raise LibcloudError(msg, driver=self)
    body = response.parse_body()
    server_hash = body.find(fixxpath(xpath='ETag', namespace=self.namespace)).text
    return server_hash