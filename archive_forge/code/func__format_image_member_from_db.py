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
def _format_image_member_from_db(self, db_image_member):
    return glance.domain.ImageMembership(id=db_image_member['id'], image_id=db_image_member['image_id'], member_id=db_image_member['member'], status=db_image_member['status'], created_at=db_image_member['created_at'], updated_at=db_image_member['updated_at'])