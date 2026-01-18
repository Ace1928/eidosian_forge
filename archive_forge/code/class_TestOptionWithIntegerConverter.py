import os
import sys
import threading
from io import BytesIO
from textwrap import dedent
import configobj
from testtools import matchers
from .. import (bedding, branch, config, controldir, diff, errors, lock,
from .. import registry as _mod_registry
from .. import tests, trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..bzr import remote
from ..transport import remote as transport_remote
from . import features, scenarios, test_server
class TestOptionWithIntegerConverter(TestOptionConverter):

    def get_option(self):
        return config.Option('foo', help='An integer.', from_unicode=config.int_from_store)

    def test_convert_invalid(self):
        opt = self.get_option()
        self.assertConvertInvalid(opt, 'forty-two')
        self.assertConvertInvalid(opt, ['a', 'list'])

    def test_convert_valid(self):
        opt = self.get_option()
        self.assertConverted(16, opt, '16')