from neutron_lib._i18n import _
from neutron_lib import exceptions
class DuplicateRecordSet(exceptions.Conflict):
    message = _('Name %(dns_name)s is duplicated in the external DNS service')