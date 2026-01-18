import collections
import json
from unittest import mock
from heatclient import exc
from heatclient.v1 import stacks
from keystoneauth1 import loading as ks_loading
from oslo_config import cfg
from heat.common import exception
from heat.common.i18n import _
from heat.common import policy
from heat.common import template_format
from heat.engine.clients.os import heat_plugin
from heat.engine import environment
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.heat import remote_stack
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template
from heat.tests import common as tests_common
from heat.tests import utils
def _create_with_remote_credential(self, credential_secret_id=None, ca_cert=None, insecure=False):
    t = template_format.parse(parent_stack_template)
    properties = t['resources']['remote_stack']['properties']
    if credential_secret_id:
        properties['context']['credential_secret_id'] = credential_secret_id
    if ca_cert:
        properties['context']['ca_cert'] = ca_cert
    if insecure:
        properties['context']['insecure'] = insecure
    t = json.dumps(t)
    self.patchobject(policy.Enforcer, 'check_is_admin')
    rsrc = self.create_remote_stack(stack_template=t)
    env = environment.get_child_environment(rsrc.stack.env, {'name': 'foo'})
    args = {'stack_name': rsrc.physical_resource_name(), 'template': template_format.parse(remote_template), 'timeout_mins': 60, 'disable_rollback': True, 'parameters': {'name': 'foo'}, 'files': self.files, 'environment': env.user_env_as_dict()}
    self.heat.stacks.create.assert_called_with(**args)
    self.assertEqual(2, len(self.heat.stacks.get.call_args_list))
    rsrc.validate()
    return rsrc