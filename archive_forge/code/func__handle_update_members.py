from unittest import mock
from glanceclient import exc
from heat.common import exception
from heat.common import template_format
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _handle_update_members(self, prop_diff):
    self.my_image.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    self.image_members.create.assert_called_once_with(self.my_image.resource_id, 'member2')
    self.image_members.delete.assert_called_once_with(self.my_image.resource_id, 'member1')