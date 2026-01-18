import urllib
from castellan.i18n import _
class ManagedObjectNotFoundError(CastellanException):
    message = _('Key not found, uuid: %(uuid)s')