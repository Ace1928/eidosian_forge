from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
class ParseTokenizeTestCase(test_base.BaseTestCase):

    @mock.patch.object(_parser, '_parse_check', lambda x: x)
    def test_tokenize(self):
        exemplar = '(( ( ((() And)) or ) (check:%(miss)s) not)) \'a-string\' "another-string"'
        expected = [('(', '('), ('(', '('), ('(', '('), ('(', '('), ('(', '('), ('(', '('), (')', ')'), ('and', 'And'), (')', ')'), (')', ')'), ('or', 'or'), (')', ')'), ('(', '('), ('check', 'check:%(miss)s'), (')', ')'), ('not', 'not'), (')', ')'), (')', ')'), ('string', 'a-string'), ('string', 'another-string')]
        result = list(_parser._parse_tokenize(exemplar))
        self.assertEqual(expected, result)