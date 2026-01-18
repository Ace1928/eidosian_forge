import typing as ty
from openstack.common import tag
from openstack import exceptions
from openstack.image import _download
from openstack import resource
from openstack import utils
def _consume_header_attrs(self, attrs):
    self.image_import_methods = []
    _image_import_methods = attrs.pop('OpenStack-image-import-methods', '')
    if _image_import_methods:
        self.image_import_methods = _image_import_methods.split(',')
    return super()._consume_header_attrs(attrs)