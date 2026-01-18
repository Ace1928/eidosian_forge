import fixtures
from unittest import mock
from oslo_config import cfg
from stevedore import driver
import testscenarios
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
from oslo_messaging import transport
class _SetDefaultsFixture(fixtures.Fixture):

    def __init__(self, set_defaults, opts, *names):
        super(_SetDefaultsFixture, self).__init__()
        self.set_defaults = set_defaults
        self.opts = opts
        self.names = names

    def setUp(self):
        super(_SetDefaultsFixture, self).setUp()

        def first(seq, default=None, key=None):
            if key is None:
                key = bool
            return next(filter(key, seq), default)

        def default(opts, name):
            return first(opts, key=lambda o: o.name == name).default
        orig_defaults = {}
        for n in self.names:
            orig_defaults[n] = default(self.opts, n)

        def restore_defaults():
            self.set_defaults(**orig_defaults)
        self.addCleanup(restore_defaults)