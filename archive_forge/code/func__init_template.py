import json
from unittest import mock
from novaclient import exceptions
from oslo_utils import excutils
from heat.common import template_format
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def _init_template(self, sg_template):
    template = template_format.parse(json.dumps(sg_template))
    self.stack = utils.parse_stack(template)
    self.sg = self.stack['ServerGroup']
    nova = mock.MagicMock()
    self.sg.client = mock.MagicMock(return_value=nova)

    class FakeNovaPlugin(object):

        @excutils.exception_filter
        def ignore_not_found(self, ex):
            if not isinstance(ex, exceptions.NotFound):
                raise ex

        def is_version_supported(self, version):
            return True

        def is_conflict(self, ex):
            return False
    self.patchobject(excutils.exception_filter, '__exit__')
    self.patchobject(self.sg, 'client_plugin', return_value=FakeNovaPlugin())
    self.sg_mgr = nova.server_groups