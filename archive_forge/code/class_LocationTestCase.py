import io
import tempfile
import textwrap
from oslotest import base
from oslo_config import cfg
class LocationTestCase(base.BaseTestCase):

    def test_user_controlled(self):
        self.assertTrue(cfg.Locations.user.is_user_controlled)
        self.assertTrue(cfg.Locations.command_line.is_user_controlled)

    def test_not_user_controlled(self):
        self.assertFalse(cfg.Locations.opt_default.is_user_controlled)
        self.assertFalse(cfg.Locations.set_default.is_user_controlled)
        self.assertFalse(cfg.Locations.set_override.is_user_controlled)