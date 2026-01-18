from neutron_lib._i18n import _
from neutron_lib import exceptions
class RouterInUse(exceptions.InUse):
    message = _('Router %(router_id)s %(reason)s')

    def __init__(self, **kwargs):
        if 'reason' not in kwargs:
            kwargs['reason'] = 'still has ports'
        super().__init__(**kwargs)