import testtools
from testtools import matchers as tt_matchers
from keystoneauth1.tests.unit import matchers as ks_matchers
class TestXMLEquals(testtools.TestCase):
    matches_xml = b'<?xml version="1.0" encoding="UTF-8"?>\n<test xmlns="https://docs.openstack.org/identity/api/v2.0">\n    <first z="0" y="1" x="2"/>\n    <second a="a" b="b"></second>\n</test>\n'
    equivalent_xml = b'<?xml version="1.0" encoding="UTF-8"?>\n<test xmlns="https://docs.openstack.org/identity/api/v2.0">\n    <second a="a" b="b"/>\n    <first z="0" y="1" x="2"></first>\n</test>\n'
    mismatches_xml = b'<?xml version="1.0" encoding="UTF-8"?>\n<test xmlns="https://docs.openstack.org/identity/api/v2.0">\n    <nope_it_fails/>\n</test>\n'
    mismatches_description = 'expected =\n<test xmlns="https://docs.openstack.org/identity/api/v2.0">\n  <first z="0" y="1" x="2"/>\n  <second a="a" b="b"/>\n</test>\n\nactual =\n<test xmlns="https://docs.openstack.org/identity/api/v2.0">\n  <nope_it_fails/>\n</test>\n'
    matches_matcher = ks_matchers.XMLEquals(matches_xml)
    matches_matches = [matches_xml, equivalent_xml]
    matches_mismatches = [mismatches_xml]
    describe_examples = [(mismatches_description, mismatches_xml, matches_matcher)]
    str_examples = [('XMLEquals(%r)' % matches_xml, matches_matcher)]

    def test_matches_match(self):
        matcher = self.matches_matcher
        matches = self.matches_matches
        mismatches = self.matches_mismatches
        for candidate in matches:
            self.assertIsNone(matcher.match(candidate))
        for candidate in mismatches:
            mismatch = matcher.match(candidate)
            self.assertIsNotNone(mismatch)
            self.assertIsNotNone(getattr(mismatch, 'describe', None))

    def test__str__(self):
        examples = self.str_examples
        for expected, matcher in examples:
            self.assertThat(matcher, tt_matchers.DocTestMatches(expected))

    def test_describe_difference(self):
        examples = self.describe_examples
        for difference, matchee, matcher in examples:
            mismatch = matcher.match(matchee)
            self.assertEqual(difference, mismatch.describe())

    def test_mismatch_details(self):
        examples = self.describe_examples
        for difference, matchee, matcher in examples:
            mismatch = matcher.match(matchee)
            details = mismatch.get_details()
            self.assertEqual(dict(details), details)