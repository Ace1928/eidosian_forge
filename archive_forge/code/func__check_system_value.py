from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
def _check_system_value(self, system):
    if system and system != 'all':
        msg = _("Only a system scope of 'all' is currently supported")
        raise exceptions.ValidationError(msg)