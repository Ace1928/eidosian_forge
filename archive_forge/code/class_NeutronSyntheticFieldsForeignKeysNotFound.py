from oslo_utils import reflection
from neutron_lib._i18n import _
from neutron_lib import exceptions
class NeutronSyntheticFieldsForeignKeysNotFound(exceptions.NeutronException):
    message = _('%(child)s does not define a foreign key for %(parent)s')