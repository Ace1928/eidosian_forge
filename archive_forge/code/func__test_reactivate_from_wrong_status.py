import glance_store as store
import webob
import glance.api.v2.image_actions as image_actions
import glance.context
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
def _test_reactivate_from_wrong_status(self, status):
    self._create_image(status)
    request = unit_test_utils.get_fake_request()
    self.assertRaises(webob.exc.HTTPForbidden, self.controller.reactivate, request, UUID1)