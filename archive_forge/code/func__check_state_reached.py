import collections
import enum
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
def _check_state_reached(self, session, expected_state, abort_on_failed_state=True):
    """Wait for the node to reach the expected state.

        :param session: The session to use for making this request.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param expected_state: The expected provisioning state to reach.
        :param abort_on_failed_state: If ``True`` (the default), abort waiting
            if the node reaches a failure state which does not match the
            expected one. Note that the failure state for ``enroll`` ->
            ``manageable`` transition is ``enroll`` again.

        :return: ``True`` if the target state is reached
        :raises: :class:`~openstack.exceptions.ResourceFailure` if the node
            reaches an error state and ``abort_on_failed_state`` is ``True``.
        """
    if self.provision_state == expected_state or (expected_state == 'available' and self.provision_state is None):
        return True
    elif not abort_on_failed_state:
        return False
    if self.provision_state.endswith(' failed') or self.provision_state == 'error':
        raise exceptions.ResourceFailure('Node %(node)s reached failure state "%(state)s"; the last error is %(error)s' % {'node': self.id, 'state': self.provision_state, 'error': self.last_error})
    elif expected_state == 'manageable' and self.provision_state == 'enroll' and self.last_error:
        raise exceptions.ResourceFailure('Node %(node)s could not reach state manageable: failed to verify management credentials; the last error is %(error)s' % {'node': self.id, 'error': self.last_error})