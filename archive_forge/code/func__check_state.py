from openstack import _log
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
def _check_state(self, ignore_error):
    if self.state == 'error' and (not ignore_error):
        raise exceptions.ResourceFailure('Introspection of node %(node)s failed: %(error)s' % {'node': self.id, 'error': self.error})
    else:
        return self.is_finished