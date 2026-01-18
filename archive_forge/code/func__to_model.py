import http.client as http
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
import webob.exc
from wsme.rest import json
from glance.api import policy
from glance.api.v2 import metadef_namespaces as namespaces
from glance.api.v2.model.metadef_namespace import Namespace
from glance.api.v2.model.metadef_property_type import PropertyType
from glance.api.v2.model.metadef_property_type import PropertyTypes
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import wsgi
import glance.db
import glance.gateway
from glance.i18n import _
import glance.notifier
import glance.schema
def _to_model(self, db_property_type):
    property_type = json.fromjson(PropertyType, db_property_type.schema)
    property_type.name = db_property_type.name
    return property_type