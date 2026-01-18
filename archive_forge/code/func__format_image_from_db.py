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
def _format_image_from_db(self, db_image, db_tags):
    properties = {}
    for prop in db_image.pop('properties'):
        if not prop['deleted']:
            properties[prop['name']] = prop['value']
    locations = [loc for loc in db_image['locations'] if loc['status'] == 'active']
    if CONF.metadata_encryption_key:
        key = CONF.metadata_encryption_key
        for location in locations:
            location['url'] = crypt.urlsafe_decrypt(key, location['url'])
    if db_image['visibility'] == 'shared' and self.context.owner != db_image['owner']:
        member = self.context.owner
    else:
        member = None
    return glance.domain.Image(image_id=db_image['id'], name=db_image['name'], status=db_image['status'], created_at=db_image['created_at'], updated_at=db_image['updated_at'], visibility=db_image['visibility'], min_disk=db_image['min_disk'], min_ram=db_image['min_ram'], protected=db_image['protected'], locations=common_utils.sort_image_locations(locations), checksum=db_image['checksum'], os_hash_algo=db_image['os_hash_algo'], os_hash_value=db_image['os_hash_value'], owner=db_image['owner'], disk_format=db_image['disk_format'], container_format=db_image['container_format'], size=db_image['size'], virtual_size=db_image['virtual_size'], extra_properties=properties, tags=db_tags, os_hidden=db_image['os_hidden'], member=member)