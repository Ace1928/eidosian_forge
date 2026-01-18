import uuid
from keystone.common import utils
from keystone import exception
from keystone.tests import unit
class FormatUrlTests(unit.BaseTestCase):

    def test_successful_formatting(self):
        url_template = 'http://server:9090/$(tenant_id)s/$(user_id)s/$(project_id)s'
        project_id = uuid.uuid4().hex
        values = {'tenant_id': 'A', 'user_id': 'B', 'project_id': project_id}
        actual_url = utils.format_url(url_template, values)
        expected_url = 'http://server:9090/A/B/%s' % (project_id,)
        self.assertEqual(expected_url, actual_url)

    def test_raises_malformed_on_missing_key(self):
        self.assertRaises(exception.MalformedEndpoint, utils.format_url, 'http://server:9090/$(tenant_id)s', {})

    def test_raises_malformed_on_wrong_type(self):
        self.assertRaises(exception.MalformedEndpoint, utils.format_url, 'http://server:9090/$(tenant_id)d', {'tenant_id': 'A'})

    def test_raises_malformed_on_incomplete_format(self):
        self.assertRaises(exception.MalformedEndpoint, utils.format_url, 'http://server:9090/$(tenant_id)', {'tenant_id': 'A'})

    def test_formatting_a_non_string(self):

        def _test(url_template):
            self.assertRaises(exception.MalformedEndpoint, utils.format_url, url_template, {})
        _test(None)
        _test(object())

    def test_substitution_with_key_not_allowed(self):
        url_template = 'http://server:9090/$(project_id)s/$(user_id)s/$(admin_token)s'
        values = {'user_id': 'B', 'admin_token': 'C'}
        self.assertRaises(exception.MalformedEndpoint, utils.format_url, url_template, values)

    def test_substitution_with_allowed_tenant_keyerror(self):
        url_template = 'http://server:9090/$(tenant_id)s/$(user_id)s'
        values = {'user_id': 'B'}
        self.assertIsNone(utils.format_url(url_template, values, silent_keyerror_failures=['tenant_id']))

    def test_substitution_with_allowed_project_keyerror(self):
        url_template = 'http://server:9090/$(project_id)s/$(user_id)s'
        values = {'user_id': 'B'}
        self.assertIsNone(utils.format_url(url_template, values, silent_keyerror_failures=['project_id']))