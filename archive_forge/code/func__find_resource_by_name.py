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
def _find_resource_by_name(self, resource, name, project_id=None, cmd_resource=None, parent_id=None, fields=None):
    if not cmd_resource:
        cmd_resource = resource
    cmd_resource_plural = self.get_resource_plural(cmd_resource)
    resource_plural = self.get_resource_plural(resource)
    obj_lister = getattr(self, 'list_%s' % cmd_resource_plural)
    params = {'name': name}
    if fields:
        params['fields'] = fields
    if project_id:
        params['tenant_id'] = project_id
    if parent_id:
        data = obj_lister(parent_id, **params)
    else:
        data = obj_lister(**params)
    collection = resource_plural
    info = data[collection]
    if len(info) > 1:
        raise exceptions.NeutronClientNoUniqueMatch(resource=resource, name=name)
    elif len(info) == 0:
        not_found_message = _("Unable to find %(resource)s with name '%(name)s'") % {'resource': resource, 'name': name}
        raise exceptions.NotFound(message=not_found_message)
    else:
        return info[0]