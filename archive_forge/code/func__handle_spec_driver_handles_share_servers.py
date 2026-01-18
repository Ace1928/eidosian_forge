from manilaclient import api_versions
from manilaclient import base
from manilaclient import exceptions
def _handle_spec_driver_handles_share_servers(self, extra_specs, spec_driver_handles_share_servers):
    """Validation and default for DHSS extra spec."""
    if spec_driver_handles_share_servers is not None:
        if 'driver_handles_share_servers' in extra_specs:
            msg = "'driver_handles_share_servers' is already set via positional argument."
            raise exceptions.CommandError(msg)
        else:
            extra_specs['driver_handles_share_servers'] = spec_driver_handles_share_servers
    else:
        msg = "'driver_handles_share_servers' is not set via positional argument."
        raise exceptions.CommandError(msg)