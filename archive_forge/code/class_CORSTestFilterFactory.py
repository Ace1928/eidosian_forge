from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
class CORSTestFilterFactory(CORSTestBase):
    """Test the CORS filter_factory method."""

    def test_filter_factory(self):
        self.config([])
        filter = cors.filter_factory(None, allowed_origin='http://valid.example.com', allow_credentials='False', max_age='', expose_headers='', allow_methods='GET', allow_headers='')
        application = filter(test_application)
        self.assertIn('http://valid.example.com', application.allowed_origins)
        config = application.allowed_origins['http://valid.example.com']
        self.assertEqual(False, config['allow_credentials'])
        self.assertIsNone(config['max_age'])
        self.assertEqual([], config['expose_headers'])
        self.assertEqual(['GET'], config['allow_methods'])
        self.assertEqual([], config['allow_headers'])

    def test_filter_factory_multiorigin(self):
        self.config([])
        filter = cors.filter_factory(None, allowed_origin='http://valid.example.com,http://other.example.com')
        application = filter(test_application)
        self.assertIn('http://valid.example.com', application.allowed_origins)
        self.assertIn('http://other.example.com', application.allowed_origins)

    def test_no_origin_fail(self):
        """Assert that a filter factory with no allowed_origin fails."""
        self.assertRaises(TypeError, cors.filter_factory, global_conf=None, allow_credentials='False', max_age='', expose_headers='', allow_methods='GET', allow_headers='')

    def test_no_origin_but_oslo_config_project(self):
        """Assert that a filter factory with oslo_config_project succeed."""
        cors.filter_factory(global_conf=None, oslo_config_project='foobar')

    def test_cor_config_sections_with_defaults(self):
        """Assert cors.* config sections with default values work."""
        self.config_fixture.load_raw_values(group='cors.subdomain')
        self.application = cors.CORS(test_application, self.config)