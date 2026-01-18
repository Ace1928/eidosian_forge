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
def _cleanup_namespace(self, namespace_repo, namespace, namespace_created):
    if namespace_created:
        try:
            namespace_obj = namespace_repo.get(namespace.namespace)
            namespace_obj.delete()
            namespace_repo.remove(namespace_obj)
            LOG.debug('Cleaned up namespace %(namespace)s ', {'namespace': namespace.namespace})
        except Exception as e:
            msg = (_LE('Failed to delete namespace %(namespace)s.Exception: %(exception)s'), {'namespace': namespace.namespace, 'exception': encodeutils.exception_to_unicode(e)})
            LOG.error(msg)