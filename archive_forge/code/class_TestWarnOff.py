import warnings
import os_service_types
from os_service_types import exc
from os_service_types.tests import base
class TestWarnOff(base.TestCase):

    def setUp(self):
        super(TestWarnOff, self).setUp()
        warnings.simplefilter('always')
        self.service_types = os_service_types.ServiceTypes()

    def test_warning_not_emitted_on_alias(self):
        with warnings.catch_warnings(record=True) as w:
            self.service_types.get_service_type('volumev2')
            self.assertEqual(0, len(w))

    def test_warning_not_emitted_on_official(self):
        with warnings.catch_warnings(record=True) as w:
            self.service_types.get_service_type('block-storage')
            self.assertEqual(0, len(w))