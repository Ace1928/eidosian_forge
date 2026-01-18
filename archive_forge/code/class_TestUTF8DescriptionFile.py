import io
import tempfile
import textwrap
import six
from six.moves import configparser
import sys
from pbr.tests import base
from pbr import util
class TestUTF8DescriptionFile(base.BaseTestCase):

    def test_utf8_description_file(self):
        _, path = tempfile.mkstemp()
        ini_template = '\n        [metadata]\n        description_file = %s\n        '
        unicode_description = u'UTF8 description: é"…-ʃŋ\'\n\n'
        ini = ini_template % path
        with io.open(path, 'w', encoding='utf8') as f:
            f.write(unicode_description)
        config = config_from_ini(ini)
        kwargs = util.setup_cfg_to_setup_kwargs(config)
        self.assertEqual(unicode_description, kwargs['long_description'])