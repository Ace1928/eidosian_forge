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
def _format_image_member_to_db(self, image_member):
    image_member = {'image_id': self.image.image_id, 'member': image_member.member_id, 'status': image_member.status, 'created_at': image_member.created_at}
    return image_member