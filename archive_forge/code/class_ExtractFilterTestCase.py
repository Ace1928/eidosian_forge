import collections
import io
import sys
from unittest import mock
import ddt
from cinderclient import api_versions
from cinderclient.apiclient import base as common_base
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient.tests.unit import utils as test_utils
from cinderclient import utils
@ddt.ddt
class ExtractFilterTestCase(test_utils.TestCase):

    @ddt.data({'content': ['key1=value1'], 'expected': {'key1': 'value1'}}, {'content': ['key1={key2:value2}'], 'expected': {'key1': {'key2': 'value2'}}}, {'content': ['key1=value1', 'key2={key22:value22}'], 'expected': {'key1': 'value1', 'key2': {'key22': 'value22'}}})
    @ddt.unpack
    def test_extract_filters(self, content, expected):
        result = shell_utils.extract_filters(content)
        self.assertEqual(expected, result)