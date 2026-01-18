from unittest import mock
from keystoneauth1 import exceptions as kc_exceptions
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.resources import stack_user
from heat.engine import scheduler
from heat.objects import resource_data as resource_data_object
from heat.tests import common
from heat.tests import utils
def _user_create(self, stack_name, project_id, user_id, resource_name='user', create_project=True, password=None):
    t = template_format.parse(user_template)
    self.stack = utils.parse_stack(t, stack_name=stack_name)
    rsrc = self.stack[resource_name]
    self.patchobject(stack_user.StackUser, 'keystone', return_value=self.fc)
    if create_project:
        self.fc.create_stack_domain_project.return_value = project_id
    else:
        self.stack.set_stack_user_project_id(project_id)
    rsrc.store()
    mock_get_id = self.patchobject(short_id, 'get_id')
    mock_get_id.return_value = 'aabbcc'
    self.fc.create_stack_domain_user.return_value = user_id
    return rsrc