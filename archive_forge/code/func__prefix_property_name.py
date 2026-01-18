import http.client as http
import urllib.parse as urlparse
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
import webob.exc
from wsme.rest import json
from glance.api import policy
from glance.api.v2.model.metadef_namespace import Namespace
from glance.api.v2.model.metadef_namespace import Namespaces
from glance.api.v2.model.metadef_object import MetadefObject
from glance.api.v2.model.metadef_property_type import PropertyType
from glance.api.v2.model.metadef_resource_type import ResourceTypeAssociation
from glance.api.v2.model.metadef_tag import MetadefTag
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import utils
from glance.common import wsgi
from glance.common import wsme_utils
import glance.db
import glance.gateway
from glance.i18n import _, _LE
import glance.notifier
import glance.schema
def _prefix_property_name(self, namespace_detail, user_resource_type):
    prefix = None
    if user_resource_type and namespace_detail.resource_type_associations:
        for resource_type in namespace_detail.resource_type_associations:
            if resource_type.name == user_resource_type:
                prefix = resource_type.prefix
                break
    if prefix:
        if namespace_detail.properties:
            new_property_dict = dict()
            for key, value in namespace_detail.properties.items():
                new_property_dict[prefix + key] = value
            namespace_detail.properties = new_property_dict
        if namespace_detail.objects:
            for object in namespace_detail.objects:
                new_object_property_dict = dict()
                for key, value in object.properties.items():
                    new_object_property_dict[prefix + key] = value
                object.properties = new_object_property_dict
                if object.required and len(object.required) > 0:
                    required = [prefix + name for name in object.required]
                    object.required = required
    return namespace_detail