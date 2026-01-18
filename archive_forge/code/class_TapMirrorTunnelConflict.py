from neutron_lib._i18n import _
from neutron_lib import exceptions as qexception
class TapMirrorTunnelConflict(qexception.Conflict):
    message = _('Tap Mirror with tunnel_id %(tunnel_id)s already exists')