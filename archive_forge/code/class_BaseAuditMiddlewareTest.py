from oslo_config import fixture as cfg_fixture
from oslo_messaging import conffixture as msg_fixture
from oslotest import createfile
import webob.dec
from keystonemiddleware import audit
from keystonemiddleware.tests.unit import utils
class BaseAuditMiddlewareTest(utils.MiddlewareTestCase):
    PROJECT_NAME = 'keystonemiddleware'

    def setUp(self):
        super(BaseAuditMiddlewareTest, self).setUp()
        self.audit_map_file_fixture = self.useFixture(createfile.CreateFileWithContent('audit', audit_map_content))
        self.cfg = self.useFixture(cfg_fixture.Config())
        self.msg = self.useFixture(msg_fixture.ConfFixture(self.cfg.conf))
        self.cfg.conf([], project=self.PROJECT_NAME)

    def create_middleware(self, cb, **kwargs):

        @webob.dec.wsgify
        def _do_cb(req):
            return cb(req)
        kwargs.setdefault('audit_map_file', self.audit_map)
        kwargs.setdefault('service_name', 'pycadf')
        return audit.AuditMiddleware(_do_cb, **kwargs)

    @property
    def audit_map(self):
        return self.audit_map_file_fixture.path

    @staticmethod
    def get_environ_header(req_type=None):
        env_headers = {'HTTP_X_SERVICE_CATALOG': '[{"endpoints_links": [],\n                            "endpoints": [{"adminURL":\n                                           "http://admin_host:8774",\n                                           "region": "RegionOne",\n                                           "publicURL":\n                                           "http://public_host:8774",\n                                           "internalURL":\n                                           "http://internal_host:8774",\n                                           "id": "resource_id"}],\n                            "type": "compute",\n                            "name": "nova"}]', 'HTTP_X_USER_ID': 'user_id', 'HTTP_X_USER_NAME': 'user_name', 'HTTP_X_AUTH_TOKEN': 'token', 'HTTP_X_PROJECT_ID': 'tenant_id', 'HTTP_X_IDENTITY_STATUS': 'Confirmed'}
        if req_type:
            env_headers['REQUEST_METHOD'] = req_type
        return env_headers