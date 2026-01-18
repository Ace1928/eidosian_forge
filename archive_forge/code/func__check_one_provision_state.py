import logging
import os
from oslo_utils import strutils
from ironicclient.common import base
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import volume_connector
from ironicclient.v1 import volume_target
def _check_one_provision_state(self, node_ident, expected_state, fail_on_unexpected_state=True, os_ironic_api_version=None, global_request_id=None):
    node = self.get(node_ident, os_ironic_api_version=os_ironic_api_version, global_request_id=global_request_id)
    if node.provision_state == expected_state:
        LOG.debug('Node %(node)s reached provision state %(state)s', {'node': node_ident, 'state': expected_state})
        return True
    if node.provision_state == 'error' or node.provision_state.endswith(' failed'):
        raise exc.StateTransitionFailed(_("Node %(node)s failed to reach state %(state)s. It's in state %(actual)s, and has error: %(error)s") % {'node': node_ident, 'state': expected_state, 'actual': node.provision_state, 'error': node.last_error})
    if fail_on_unexpected_state and (not node.target_provision_state):
        raise exc.StateTransitionFailed(_("Node %(node)s failed to reach state %(state)s. It's in unexpected stable state %(actual)s") % {'node': node_ident, 'state': expected_state, 'actual': node.provision_state})
    LOG.debug('Still waiting for node %(node)s to reach state %(state)s, the current state is %(actual)s', {'node': node_ident, 'state': expected_state, 'actual': node.provision_state})
    return False