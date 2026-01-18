import datetime
import os
from unittest import mock
import glance_store as store_api
from oslo_config import cfg
from glance.async_.flows._internal_plugins import copy_image
from glance.async_.flows import api_image_import
import glance.common.exception as exception
from glance import domain
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def _db_fixture(id, **kwargs):
    obj = {'id': id, 'name': None, 'visibility': 'shared', 'properties': {}, 'checksum': None, 'os_hash_algo': FAKEHASHALGO, 'os_hash_value': None, 'owner': None, 'status': 'queued', 'tags': [], 'size': None, 'virtual_size': None, 'locations': [], 'protected': False, 'disk_format': None, 'container_format': None, 'deleted': False, 'min_ram': None, 'min_disk': None}
    obj.update(kwargs)
    return obj