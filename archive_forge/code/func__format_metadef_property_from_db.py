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
def _format_metadef_property_from_db(self, property, namespace_entity):
    return glance.domain.MetadefProperty(namespace=namespace_entity, property_id=property['id'], name=property['name'], schema=property['json_schema'])