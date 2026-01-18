from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
def _validate_interface(self, interface):
    if interface is not None and interface not in VALID_INTERFACES:
        msg = _('"interface" must be one of: %s')
        msg %= ', '.join(VALID_INTERFACES)
        raise exceptions.ValidationError(msg)