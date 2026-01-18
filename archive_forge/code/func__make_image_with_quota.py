import copy
import fixtures
from unittest import mock
from unittest.mock import patch
import uuid
from oslo_limit import exception as ol_exc
from oslo_utils import encodeutils
from oslo_utils import units
from glance.common import exception
from glance.common import store_utils
import glance.quota
from glance.quota import keystone as ks_quota
from glance.tests.unit import fixtures as glance_fixtures
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils as test_utils
def _make_image_with_quota(self, image_size=10, location_count=2):
    quota = image_size * location_count
    self.config(user_storage_quota=str(quota))
    return self._get_image(image_size=image_size, location_count=location_count)