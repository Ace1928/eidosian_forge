from oslo_utils import excutils
from neutron_lib._i18n import _
class UnsupportedMinSubnetPoolPrefix(BadRequest):
    message = _("Prefix '%(prefix)s' not supported in IPv%(version)s pool.")