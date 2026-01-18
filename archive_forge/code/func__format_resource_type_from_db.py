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
def _format_resource_type_from_db(self, resource_type, namespace):
    return glance.domain.MetadefResourceType(namespace=namespace, name=resource_type['name'], prefix=resource_type['prefix'], properties_target=resource_type['properties_target'], created_at=resource_type['created_at'], updated_at=resource_type['updated_at'])