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
class BaseS3StorageDriver(StorageDriver):
    name = 'Amazon S3 (standard)'
    website = 'http://aws.amazon.com/s3/'
    connectionCls = BaseS3Connection
    hash_type = 'md5'
    supports_chunked_encoding = False
    supports_s3_multipart_upload = True
    ex_location_name = ''
    namespace = NAMESPACE
    http_vendor_prefix = 'x-amz'

    def iterate_containers(self):
        response = self.connection.request('/')
        if response.status == httplib.OK:
            containers = self._to_containers(obj=response.object, xpath='Buckets/Bucket')
            return containers
        raise LibcloudError('Unexpected status code: %s' % response.status, driver=self)

    def iterate_container_objects(self, container, prefix=None, ex_prefix=None):
        """
        Return a generator of objects for the given container.

        :param container: Container instance
        :type container: :class:`Container`

        :param prefix: Only return objects starting with prefix
        :type prefix: ``str``

        :param ex_prefix: Only return objects starting with ex_prefix
        :type ex_prefix: ``str``

        :return: A generator of Object instances.
        :rtype: ``generator`` of :class:`Object`
        """
        prefix = self._normalize_prefix_argument(prefix, ex_prefix)
        params = {}
        if prefix:
            params['prefix'] = prefix
        last_key = None
        exhausted = False
        container_path = self._get_container_path(container)
        while not exhausted:
            if last_key:
                params['marker'] = last_key
            response = self.connection.request(container_path, params=params)
            if response.status != httplib.OK:
                raise LibcloudError('Unexpected status code: %s' % response.status, driver=self)
            objects = self._to_objs(obj=response.object, xpath='Contents', container=container)
            is_truncated = response.object.findtext(fixxpath(xpath='IsTruncated', namespace=self.namespace)).lower()
            exhausted = is_truncated == 'false'
            last_key = None
            for obj in objects:
                last_key = obj.name
                yield obj

    def get_container(self, container_name):
        try:
            response = self.connection.request('/%s' % container_name, method='HEAD')
            if response.status == httplib.NOT_FOUND:
                raise ContainerDoesNotExistError(value=None, driver=self, container_name=container_name)
        except InvalidCredsError:
            pass
        return Container(name=container_name, extra=None, driver=self)

    def get_object(self, container_name, object_name):
        container = self.get_container(container_name=container_name)
        object_path = self._get_object_path(container, object_name)
        response = self.connection.request(object_path, method='HEAD')
        if response.status == httplib.OK:
            obj = self._headers_to_object(object_name=object_name, container=container, headers=response.headers)
            return obj
        raise ObjectDoesNotExistError(value=None, driver=self, object_name=object_name)

    def _get_container_path(self, container):
        """
        Return a container path

        :param container: Container instance
        :type  container: :class:`Container`

        :return: A path for this container.
        :rtype: ``str``
        """
        return '/%s' % container.name

    def _get_object_path(self, container, object_name):
        """
        Return an object's CDN path.

        :param container: Container instance
        :type  container: :class:`Container`

        :param object_name: Object name
        :type  object_name: :class:`str`

        :return: A  path for this object.
        :rtype: ``str``
        """
        container_url = self._get_container_path(container)
        object_name_cleaned = self._clean_object_name(object_name)
        object_path = '{}/{}'.format(container_url, object_name_cleaned)
        return object_path

    def create_container(self, container_name):
        if self.ex_location_name:
            root = Element('CreateBucketConfiguration')
            child = SubElement(root, 'LocationConstraint')
            child.text = self.ex_location_name
            data = tostring(root)
        else:
            data = ''
        response = self.connection.request('/%s' % container_name, data=data, method='PUT')
        if response.status == httplib.OK:
            container = Container(name=container_name, extra=None, driver=self)
            return container
        elif response.status == httplib.CONFLICT:
            if 'BucketAlreadyOwnedByYou' in response.body:
                raise ContainerAlreadyExistsError(value='Container with this name already exists. The name be unique among all the containers in the system.', container_name=container_name, driver=self)
            raise InvalidContainerNameError(value='Container with this name already exists. The name must be unique among all the containers in the system.', container_name=container_name, driver=self)
        elif response.status == httplib.BAD_REQUEST:
            raise ContainerError(value='Bad request when creating container: %s' % response.body, container_name=container_name, driver=self)
        raise LibcloudError('Unexpected status code: %s' % response.status, driver=self)

    def delete_container(self, container):
        response = self.connection.request('/%s' % container.name, method='DELETE')
        if response.status == httplib.NO_CONTENT:
            return True
        elif response.status == httplib.CONFLICT:
            raise ContainerIsNotEmptyError(value='Container must be empty before it can be deleted.', container_name=container.name, driver=self)
        elif response.status == httplib.NOT_FOUND:
            raise ContainerDoesNotExistError(value=None, driver=self, container_name=container.name)
        return False

    def download_object(self, obj, destination_path, overwrite_existing=False, delete_on_failure=True):
        obj_path = self._get_object_path(obj.container, obj.name)
        response = self.connection.request(obj_path, method='GET', raw=True)
        return self._get_object(obj=obj, callback=self._save_object, response=response, callback_kwargs={'obj': obj, 'response': response.response, 'destination_path': destination_path, 'overwrite_existing': overwrite_existing, 'delete_on_failure': delete_on_failure}, success_status_code=httplib.OK)

    def download_object_as_stream(self, obj, chunk_size=None):
        obj_path = self._get_object_path(obj.container, obj.name)
        response = self.connection.request(obj_path, method='GET', stream=True, raw=True)
        return self._get_object(obj=obj, callback=read_in_chunks, response=response, callback_kwargs={'iterator': response.iter_content(CHUNK_SIZE), 'chunk_size': chunk_size}, success_status_code=httplib.OK)

    def download_object_range(self, obj, destination_path, start_bytes, end_bytes=None, overwrite_existing=False, delete_on_failure=True):
        self._validate_start_and_end_bytes(start_bytes=start_bytes, end_bytes=end_bytes)
        obj_path = self._get_object_path(obj.container, obj.name)
        headers = {'Range': self._get_standard_range_str(start_bytes, end_bytes)}
        response = self.connection.request(obj_path, method='GET', headers=headers, raw=True)
        return self._get_object(obj=obj, callback=self._save_object, response=response, callback_kwargs={'obj': obj, 'response': response.response, 'destination_path': destination_path, 'overwrite_existing': overwrite_existing, 'delete_on_failure': delete_on_failure, 'partial_download': True}, success_status_code=httplib.PARTIAL_CONTENT)

    def download_object_range_as_stream(self, obj, start_bytes, end_bytes=None, chunk_size=None):
        self._validate_start_and_end_bytes(start_bytes=start_bytes, end_bytes=end_bytes)
        obj_path = self._get_object_path(obj.container, obj.name)
        headers = {'Range': self._get_standard_range_str(start_bytes, end_bytes)}
        response = self.connection.request(obj_path, method='GET', headers=headers, stream=True, raw=True)
        return self._get_object(obj=obj, callback=read_in_chunks, response=response, callback_kwargs={'iterator': response.iter_content(CHUNK_SIZE), 'chunk_size': chunk_size}, success_status_code=httplib.PARTIAL_CONTENT)

    def upload_object(self, file_path, container, object_name, extra=None, verify_hash=True, headers=None, ex_storage_class=None):
        """
        @inherits: :class:`StorageDriver.upload_object`

        :param ex_storage_class: Storage class
        :type ex_storage_class: ``str``
        """
        return self._put_object(container=container, object_name=object_name, extra=extra, file_path=file_path, verify_hash=verify_hash, headers=headers, storage_class=ex_storage_class)

    def _initiate_multipart(self, container, object_name, headers=None):
        """
        Initiates a multipart upload to S3

        :param container: The destination container
        :type container: :class:`Container`

        :param object_name: The name of the object which we are uploading
        :type object_name: ``str``

        :keyword headers: Additional headers to send with the request
        :type headers: ``dict``

        :return: The id of the newly created multipart upload
        :rtype: ``str``
        """
        headers = headers or {}
        request_path = self._get_object_path(container, object_name)
        params = {'uploads': ''}
        response = self.connection.request(request_path, method='POST', headers=headers, params=params)
        if response.status != httplib.OK:
            raise LibcloudError('Error initiating multipart upload', driver=self)
        return findtext(element=response.object, xpath='UploadId', namespace=self.namespace)

    def _upload_multipart_chunks(self, container, object_name, upload_id, stream, calculate_hash=True):
        """
        Uploads data from an iterator in fixed sized chunks to S3

        :param container: The destination container
        :type container: :class:`Container`

        :param object_name: The name of the object which we are uploading
        :type object_name: ``str``

        :param upload_id: The upload id allocated for this multipart upload
        :type upload_id: ``str``

        :param stream: The generator for fetching the upload data
        :type stream: ``generator``

        :keyword calculate_hash: Indicates if we must calculate the data hash
        :type calculate_hash: ``bool``

        :return: A tuple of (chunk info, checksum, bytes transferred)
        :rtype: ``tuple``
        """
        data_hash = None
        if calculate_hash:
            data_hash = self._get_hash_function()
        bytes_transferred = 0
        count = 1
        chunks = []
        params = {'uploadId': upload_id}
        request_path = self._get_object_path(container, object_name)
        for data in read_in_chunks(stream, chunk_size=CHUNK_SIZE, fill_size=True, yield_empty=True):
            bytes_transferred += len(data)
            if calculate_hash:
                data_hash.update(data)
            chunk_hash = self._get_hash_function()
            chunk_hash.update(data)
            chunk_hash = base64.b64encode(chunk_hash.digest()).decode('utf-8')
            headers = {'Content-Length': len(data), 'Content-MD5': chunk_hash}
            params['partNumber'] = count
            resp = self.connection.request(request_path, method='PUT', data=data, headers=headers, params=params)
            if resp.status != httplib.OK:
                raise LibcloudError('Error uploading chunk', driver=self)
            server_hash = resp.headers['etag'].replace('"', '')
            chunks.append((count, server_hash))
            count += 1
        if calculate_hash:
            data_hash = data_hash.hexdigest()
        return (chunks, data_hash, bytes_transferred)

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

    def _abort_multipart(self, container, object_name, upload_id):
        """
        Aborts an already initiated multipart upload

        :param container: The destination container
        :type container: :class:`Container`

        :param object_name: The name of the object which we are uploading
        :type object_name: ``str``

        :param upload_id: The upload id allocated for this multipart upload
        :type upload_id: ``str``
        """
        params = {'uploadId': upload_id}
        request_path = self._get_object_path(container, object_name)
        resp = self.connection.request(request_path, method='DELETE', params=params)
        if resp.status != httplib.NO_CONTENT:
            raise LibcloudError('Error in multipart abort. status_code=%d' % resp.status, driver=self)

    def upload_object_via_stream(self, iterator, container, object_name, extra=None, headers=None, ex_storage_class=None):
        """
        @inherits: :class:`StorageDriver.upload_object_via_stream`

        :param ex_storage_class: Storage class
        :type ex_storage_class: ``str``
        """
        method = 'PUT'
        params = None
        if self.supports_s3_multipart_upload:
            return self._put_object_multipart(container=container, object_name=object_name, extra=extra, stream=iterator, verify_hash=False, headers=headers, storage_class=ex_storage_class)
        return self._put_object(container=container, object_name=object_name, extra=extra, method=method, query_args=params, stream=iterator, verify_hash=False, headers=headers, storage_class=ex_storage_class)

    def delete_object(self, obj):
        object_path = self._get_object_path(obj.container, obj.name)
        response = self.connection.request(object_path, method='DELETE')
        if response.status == httplib.NO_CONTENT:
            return True
        elif response.status == httplib.NOT_FOUND:
            raise ObjectDoesNotExistError(value=None, driver=self, object_name=obj.name)
        return False

    def ex_iterate_multipart_uploads(self, container, prefix=None, delimiter=None):
        """
        Extension method for listing all in-progress S3 multipart uploads.

        Each multipart upload which has not been committed or aborted is
        considered in-progress.

        :param container: The container holding the uploads
        :type container: :class:`Container`

        :keyword prefix: Print only uploads of objects with this prefix
        :type prefix: ``str``

        :keyword delimiter: The object/key names are grouped based on
            being split by this delimiter
        :type delimiter: ``str``

        :return: A generator of S3MultipartUpload instances.
        :rtype: ``generator`` of :class:`S3MultipartUpload`
        """
        if not self.supports_s3_multipart_upload:
            raise LibcloudError('Feature not supported', driver=self)
        request_path = self._get_container_path(container)
        params = {'max-uploads': RESPONSES_PER_REQUEST, 'uploads': ''}
        if prefix:
            params['prefix'] = prefix
        if delimiter:
            params['delimiter'] = delimiter

        def finder(node, text):
            return node.findtext(fixxpath(xpath=text, namespace=self.namespace))
        while True:
            response = self.connection.request(request_path, params=params)
            if response.status != httplib.OK:
                raise LibcloudError('Error fetching multipart uploads. Got code: %s' % response.status, driver=self)
            body = response.parse_body()
            for node in body.findall(fixxpath(xpath='Upload', namespace=self.namespace)):
                initiator = node.find(fixxpath(xpath='Initiator', namespace=self.namespace))
                owner = node.find(fixxpath(xpath='Owner', namespace=self.namespace))
                key = finder(node, 'Key')
                upload_id = finder(node, 'UploadId')
                created_at = finder(node, 'Initiated')
                initiator = finder(initiator, 'DisplayName')
                owner = finder(owner, 'DisplayName')
                yield S3MultipartUpload(key, upload_id, created_at, initiator, owner)
            is_truncated = body.findtext(fixxpath(xpath='IsTruncated', namespace=self.namespace))
            if is_truncated.lower() == 'false':
                break
            upload_marker = body.findtext(fixxpath(xpath='NextUploadIdMarker', namespace=self.namespace))
            key_marker = body.findtext(fixxpath(xpath='NextKeyMarker', namespace=self.namespace))
            params['key-marker'] = key_marker
            params['upload-id-marker'] = upload_marker

    def ex_cleanup_all_multipart_uploads(self, container, prefix=None):
        """
        Extension method for removing all partially completed S3 multipart
        uploads.

        :param container: The container holding the uploads
        :type container: :class:`Container`

        :keyword prefix: Delete only uploads of objects with this prefix
        :type prefix: ``str``
        """
        for upload in self.ex_iterate_multipart_uploads(container, prefix, delimiter=None):
            self._abort_multipart(container, upload.key, upload.id)

    def _clean_object_name(self, name):
        name = urlquote(name, safe='/~')
        return name

    def _put_object(self, container, object_name, method='PUT', query_args=None, extra=None, file_path=None, stream=None, verify_hash=True, storage_class=None, headers=None):
        headers = headers or {}
        extra = extra or {}
        headers.update(self._to_storage_class_headers(storage_class))
        content_type = extra.get('content_type', None)
        meta_data = extra.get('meta_data', None)
        acl = extra.get('acl', None)
        if meta_data:
            for key, value in list(meta_data.items()):
                key = self.http_vendor_prefix + '-meta-%s' % key
                headers[key] = value
        if acl:
            headers[self.http_vendor_prefix + '-acl'] = acl
        request_path = self._get_object_path(container, object_name)
        if query_args:
            request_path = '?'.join((request_path, query_args))
        result_dict = self._upload_object(object_name=object_name, content_type=content_type, request_path=request_path, request_method=method, headers=headers, file_path=file_path, stream=stream)
        response = result_dict['response']
        bytes_transferred = result_dict['bytes_transferred']
        headers = response.headers
        response = response
        server_hash = headers.get('etag', '').replace('"', '')
        server_side_encryption = headers.get('x-amz-server-side-encryption', None)
        aws_kms_encryption = server_side_encryption == 'aws:kms'
        hash_matches = result_dict['data_hash'] == server_hash
        if verify_hash and (not aws_kms_encryption) and (not hash_matches):
            raise ObjectHashMismatchError(value='MD5 hash {} checksum does not match {}'.format(server_hash, result_dict['data_hash']), object_name=object_name, driver=self)
        elif response.status == httplib.OK:
            obj = Object(name=object_name, size=bytes_transferred, hash=server_hash, extra={'acl': acl}, meta_data=meta_data, container=container, driver=self)
            return obj
        else:
            raise LibcloudError('Unexpected status code, status_code=%s' % response.status, driver=self)

    def _put_object_multipart(self, container, object_name, stream, extra=None, verify_hash=False, headers=None, storage_class=None):
        """
        Uploads an object using the S3 multipart algorithm.

        :param container: The destination container
        :type container: :class:`Container`

        :param object_name: The name of the object which we are uploading
        :type object_name: ``str``

        :param stream: The generator for fetching the upload data
        :type stream: ``generator``

        :keyword verify_hash: Indicates if we must calculate the data hash
        :type verify_hash: ``bool``

        :keyword extra: Additional options
        :type extra: ``dict``

        :keyword headers: Additional headers
        :type headers: ``dict``

        :keyword storage_class: The name of the S3 object's storage class
        :type extra: ``str``

        :return: The uploaded object
        :rtype: :class:`Object`
        """
        headers = headers or {}
        extra = extra or {}
        headers.update(self._to_storage_class_headers(storage_class))
        content_type = extra.get('content_type', None)
        meta_data = extra.get('meta_data', None)
        acl = extra.get('acl', None)
        headers['Content-Type'] = self._determine_content_type(content_type, object_name)
        if meta_data:
            for key, value in list(meta_data.items()):
                key = self.http_vendor_prefix + '-meta-%s' % key
                headers[key] = value
        if acl:
            headers[self.http_vendor_prefix + '-acl'] = acl
        upload_id = self._initiate_multipart(container, object_name, headers=headers)
        try:
            result = self._upload_multipart_chunks(container, object_name, upload_id, stream, calculate_hash=verify_hash)
            chunks, data_hash, bytes_transferred = result
            etag = self._commit_multipart(container, object_name, upload_id, chunks)
        except Exception:
            self._abort_multipart(container, object_name, upload_id)
            raise
        return Object(name=object_name, size=bytes_transferred, hash=etag, extra={'acl': acl}, meta_data=meta_data, container=container, driver=self)

    def _to_storage_class_headers(self, storage_class):
        """
        Generates request headers given a storage class name.

        :keyword storage_class: The name of the S3 object's storage class
        :type extra: ``str``

        :return: Headers to include in a request
        :rtype: :dict:
        """
        headers = {}
        storage_class = storage_class or 'standard'
        if storage_class not in ['standard', 'reduced_redundancy', 'standard_ia', 'onezone_ia', 'intelligent_tiering', 'glacier', 'deep_archive', 'glacier_ir']:
            raise ValueError('Invalid storage class value: %s' % storage_class)
        key = self.http_vendor_prefix + '-storage-class'
        headers[key] = storage_class.upper()
        return headers

    def _to_containers(self, obj, xpath):
        for element in obj.findall(fixxpath(xpath=xpath, namespace=self.namespace)):
            yield self._to_container(element)

    def _to_objs(self, obj, xpath, container):
        return [self._to_obj(element, container) for element in obj.findall(fixxpath(xpath=xpath, namespace=self.namespace))]

    def _to_container(self, element):
        extra = {'creation_date': findtext(element=element, xpath='CreationDate', namespace=self.namespace)}
        container = Container(name=findtext(element=element, xpath='Name', namespace=self.namespace), extra=extra, driver=self)
        return container

    def _get_content_length_from_headers(self, headers: Dict[str, str]) -> Optional[int]:
        """
        Prase object size from the provided response headers.
        """
        content_length = headers.get('content-length', None)
        return content_length

    def _headers_to_object(self, object_name, container, headers):
        hash = headers.get('etag', '').replace('"', '')
        extra = {}
        if 'content-type' in headers:
            extra['content_type'] = headers['content-type']
        if 'etag' in headers:
            extra['etag'] = headers['etag']
        meta_data = {}
        if 'content-encoding' in headers:
            extra['content_encoding'] = headers['content-encoding']
        if 'last-modified' in headers:
            extra['last_modified'] = headers['last-modified']
        for key, value in headers.items():
            if not key.lower().startswith(self.http_vendor_prefix + '-meta-'):
                continue
            key = key.replace(self.http_vendor_prefix + '-meta-', '')
            meta_data[key] = value
        content_length = self._get_content_length_from_headers(headers=headers)
        if content_length is None:
            raise KeyError('Can not deduce object size from headers for object %s' % object_name)
        obj = Object(name=object_name, size=int(content_length), hash=hash or None, extra=extra, meta_data=meta_data, container=container, driver=self)
        return obj

    def _to_obj(self, element, container):
        owner_id = findtext(element=element, xpath='Owner/ID', namespace=self.namespace)
        owner_display_name = findtext(element=element, xpath='Owner/DisplayName', namespace=self.namespace)
        meta_data = {'owner': {'id': owner_id, 'display_name': owner_display_name}}
        last_modified = findtext(element=element, xpath='LastModified', namespace=self.namespace)
        extra = {'last_modified': last_modified}
        obj = Object(name=findtext(element=element, xpath='Key', namespace=self.namespace), size=int(findtext(element=element, xpath='Size', namespace=self.namespace)), hash=findtext(element=element, xpath='ETag', namespace=self.namespace).replace('"', ''), extra=extra, meta_data=meta_data, container=container, driver=self)
        return obj