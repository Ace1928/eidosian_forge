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
def _format_image_to_db(self, image):
    locations = image.locations
    if CONF.metadata_encryption_key:
        key = CONF.metadata_encryption_key
        ld = []
        for loc in locations:
            url = crypt.urlsafe_encrypt(key, loc['url'])
            ld.append({'url': url, 'metadata': loc['metadata'], 'status': loc['status'], 'id': loc.get('id')})
        locations = ld
    return {'id': image.image_id, 'name': image.name, 'status': image.status, 'created_at': image.created_at, 'min_disk': image.min_disk, 'min_ram': image.min_ram, 'protected': image.protected, 'locations': locations, 'checksum': image.checksum, 'os_hash_algo': image.os_hash_algo, 'os_hash_value': image.os_hash_value, 'owner': image.owner, 'disk_format': image.disk_format, 'container_format': image.container_format, 'size': image.size, 'virtual_size': image.virtual_size, 'visibility': image.visibility, 'properties': dict(image.extra_properties), 'os_hidden': image.os_hidden}