from openstack import exceptions
from openstack.tests.functional import base
def _create_application_credential_with_access_rule(self):
    """create application credential with access_rule."""
    app_cred = self.conn.identity.create_application_credential(user=self.user_id, name='app_cred', access_rules=[{'path': '/v2.0/metrics', 'service': 'monitoring', 'method': 'GET'}])
    self.addCleanup(self.conn.identity.delete_application_credential, self.user_id, app_cred['id'])
    return app_cred