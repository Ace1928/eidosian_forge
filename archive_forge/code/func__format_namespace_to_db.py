from oslo_config import cfg
from oslo_utils import importutils
from wsme.rest import json
from glance.api.v2.model.metadef_property_type import PropertyType
from glance.common import crypt
from glance.common import exception
from glance.common import utils as common_utils
import glance.domain
import glance.domain.proxy
from glance.i18n import _
def _format_namespace_to_db(self, namespace_obj):
    namespace = {'namespace': namespace_obj.namespace, 'display_name': namespace_obj.display_name, 'description': namespace_obj.description, 'visibility': namespace_obj.visibility, 'protected': namespace_obj.protected, 'owner': namespace_obj.owner}
    return namespace