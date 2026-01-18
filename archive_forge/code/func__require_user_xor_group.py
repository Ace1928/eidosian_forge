from debtcollector import removals
from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
def _require_user_xor_group(self, user, group):
    if user and group:
        msg = _('Specify either a user or group, not both')
        raise exceptions.ValidationError(msg)
    elif not (user or group):
        msg = _('Must specify either a user or group')
        raise exceptions.ValidationError(msg)