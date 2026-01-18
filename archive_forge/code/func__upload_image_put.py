import os
import time
import typing as ty
import warnings
from openstack import exceptions
from openstack.image.v2 import cache as _cache
from openstack.image.v2 import image as _image
from openstack.image.v2 import member as _member
from openstack.image.v2 import metadef_namespace as _metadef_namespace
from openstack.image.v2 import metadef_object as _metadef_object
from openstack.image.v2 import metadef_property as _metadef_property
from openstack.image.v2 import metadef_resource_type as _metadef_resource_type
from openstack.image.v2 import metadef_schema as _metadef_schema
from openstack.image.v2 import schema as _schema
from openstack.image.v2 import service_info as _si
from openstack.image.v2 import task as _task
from openstack import proxy
from openstack import resource
from openstack import utils
from openstack import warnings as os_warnings
def _upload_image_put(self, name, filename, data, meta, validate_checksum, use_import=False, stores=None, all_stores=None, all_stores_must_succeed=None, **image_kwargs):
    if stores or all_stores or all_stores_must_succeed:
        use_import = True
    if filename and (not data):
        image_data = open(filename, 'rb')
    else:
        image_data = data
    properties = image_kwargs.pop('properties', {})
    image_kwargs.update(self._make_v2_image_params(meta, properties))
    image_kwargs['name'] = name
    image = self._create(_image.Image, **image_kwargs)
    image.data = image_data
    supports_import = image.image_import_methods and 'glance-direct' in image.image_import_methods
    if use_import and (not supports_import):
        raise exceptions.SDKException('Importing image was requested but the cloud does not support the image import method.')
    try:
        if not use_import:
            response = image.upload(self)
            exceptions.raise_from_response(response)
        if use_import:
            image.stage(self)
            image.import_image(self)
        md5 = image_kwargs.get(self._IMAGE_MD5_KEY)
        sha256 = image_kwargs.get(self._IMAGE_SHA256_KEY)
        if validate_checksum and (md5 or sha256):
            data = image.fetch(self)
            checksum = data.get('checksum')
            if checksum:
                valid = checksum == md5 or checksum == sha256
                if not valid:
                    raise Exception('Image checksum verification failed')
    except Exception:
        self.log.debug('Deleting failed upload of image %s', name)
        self.delete_image(image.id)
        raise
    return image