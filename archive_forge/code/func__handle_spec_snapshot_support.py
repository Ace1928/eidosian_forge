from manilaclient import api_versions
from manilaclient import base
from manilaclient import exceptions
def _handle_spec_snapshot_support(self, extra_specs, spec_snapshot_support, set_default=False):
    """Validation and default for snapshot extra spec."""
    if spec_snapshot_support is not None:
        if 'snapshot_support' in extra_specs:
            msg = "'snapshot_support' extra spec is provided twice."
            raise exceptions.CommandError(msg)
        else:
            extra_specs['snapshot_support'] = spec_snapshot_support
    elif 'snapshot_support' not in extra_specs and set_default:
        extra_specs['snapshot_support'] = True