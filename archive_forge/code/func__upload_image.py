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
def _upload_image(self, name, *, filename=None, data=None, meta=None, wait=False, timeout=None, validate_checksum=True, use_import=False, stores=None, all_stores=None, all_stores_must_succeed=None, **kwargs):
    if 'is_public' in kwargs['properties']:
        is_public = kwargs['properties'].pop('is_public')
        if is_public:
            kwargs['visibility'] = 'public'
        else:
            kwargs['visibility'] = 'private'
    try:
        if self._connection.image_api_use_tasks:
            if use_import:
                raise exceptions.SDKException('The Glance Task API and Import API are mutually exclusive. Either disable image_api_use_tasks in config, or do not request using import')
            return self._upload_image_task(name, filename, data=data, meta=meta, wait=wait, timeout=timeout, **kwargs)
        else:
            return self._upload_image_put(name, filename, data=data, meta=meta, validate_checksum=validate_checksum, use_import=use_import, stores=stores, all_stores=all_stores, all_stores_must_succeed=all_stores_must_succeed, **kwargs)
    except exceptions.SDKException:
        self.log.debug('Image creation failed', exc_info=True)
        raise
    except Exception as e:
        raise exceptions.SDKException('Image creation failed: {message}'.format(message=str(e)))