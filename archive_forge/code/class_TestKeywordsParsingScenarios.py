import io
import tempfile
import textwrap
import six
from six.moves import configparser
import sys
from pbr.tests import base
from pbr import util
class TestKeywordsParsingScenarios(base.BaseTestCase):
    scenarios = [('keywords_list', {'config_text': '\n                [metadata]\n                keywords =\n                    one\n                    two\n                    three\n                ', 'expected_keywords': ['one', 'two', 'three']}), ('inline_keywords', {'config_text': '\n                [metadata]\n                keywords = one, two, three\n                ', 'expected_keywords': ['one, two, three']})]

    def test_keywords_parsing(self):
        config = config_from_ini(self.config_text)
        kwargs = util.setup_cfg_to_setup_kwargs(config)
        self.assertEqual(self.expected_keywords, kwargs['keywords'])