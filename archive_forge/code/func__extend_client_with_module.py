import inspect
import itertools
import logging
import re
import time
import urllib.parse as urlparse
import debtcollector.renames
from keystoneauth1 import exceptions as ksa_exc
import requests
from neutronclient._i18n import _
from neutronclient import client
from neutronclient.common import exceptions
from neutronclient.common import extension as client_extension
from neutronclient.common import serializer
from neutronclient.common import utils
def _extend_client_with_module(self, module, version):
    classes = inspect.getmembers(module, inspect.isclass)
    for cls_name, cls in classes:
        if hasattr(cls, 'versions'):
            if version not in cls.versions:
                continue
        parent_resource = getattr(cls, 'parent_resource', None)
        if issubclass(cls, client_extension.ClientExtensionList):
            self.extend_list(cls.resource_plural, cls.object_path, parent_resource)
        elif issubclass(cls, client_extension.ClientExtensionCreate):
            self.extend_create(cls.resource, cls.object_path, parent_resource)
        elif issubclass(cls, client_extension.ClientExtensionUpdate):
            self.extend_update(cls.resource, cls.resource_path, parent_resource)
        elif issubclass(cls, client_extension.ClientExtensionDelete):
            self.extend_delete(cls.resource, cls.resource_path, parent_resource)
        elif issubclass(cls, client_extension.ClientExtensionShow):
            self.extend_show(cls.resource, cls.resource_path, parent_resource)
        elif issubclass(cls, client_extension.NeutronClientExtension):
            setattr(self, '%s_path' % cls.resource_plural, cls.object_path)
            setattr(self, '%s_path' % cls.resource, cls.resource_path)
            self.EXTED_PLURALS.update({cls.resource_plural: cls.resource})