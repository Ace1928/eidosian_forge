from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
def _do_reset_state(self, snapshot, state, action_name='reset_status'):
    """Update the specified share snapshot with the provided state."""
    return self._action(action_name, snapshot, {'status': state})