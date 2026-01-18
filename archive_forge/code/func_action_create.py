import os
import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
def action_create(self, act_def, admin=True, scope='private', namespace=''):
    params = '{0}'.format(act_def)
    if scope == 'public':
        params += ' --public'
    if namespace:
        params += ' --namespace ' + namespace
    acts = self.mistral_cli(admin, 'action-create', params=params)
    for action in acts:
        self.addCleanup(self.mistral_cli, admin, 'action-delete', params=action['Name'])
    return acts