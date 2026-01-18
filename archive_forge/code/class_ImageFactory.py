from collections import abc
import datetime
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import importutils
from glance.common import exception
from glance.common import timeutils
from glance.i18n import _, _LE, _LI, _LW
class ImageFactory(object):
    _readonly_properties = ['created_at', 'updated_at', 'status', 'checksum', 'os_hash_algo', 'os_hash_value', 'size', 'virtual_size']
    _reserved_properties = ['owner', 'locations', 'deleted', 'deleted_at', 'direct_url', 'self', 'file', 'schema']

    def _check_readonly(self, kwargs):
        for key in self._readonly_properties:
            if key in kwargs:
                raise exception.ReadonlyProperty(property=key)

    def _check_unexpected(self, kwargs):
        if kwargs:
            msg = _('new_image() got unexpected keywords %s')
            raise TypeError(msg % kwargs.keys())

    def _check_reserved(self, properties):
        if properties is not None:
            for key in self._reserved_properties:
                if key in properties:
                    raise exception.ReservedProperty(property=key)

    def new_image(self, image_id=None, name=None, visibility='shared', min_disk=0, min_ram=0, protected=False, owner=None, disk_format=None, container_format=None, extra_properties=None, tags=None, os_hidden=False, **other_args):
        extra_properties = extra_properties or {}
        self._check_readonly(other_args)
        self._check_unexpected(other_args)
        self._check_reserved(extra_properties)
        if image_id is None:
            image_id = str(uuid.uuid4())
        created_at = timeutils.utcnow()
        updated_at = created_at
        status = 'queued'
        return Image(image_id=image_id, name=name, status=status, created_at=created_at, updated_at=updated_at, visibility=visibility, min_disk=min_disk, min_ram=min_ram, protected=protected, owner=owner, disk_format=disk_format, container_format=container_format, os_hidden=os_hidden, extra_properties=extra_properties, tags=tags or [])