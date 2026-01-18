import urllib
from castellan.i18n import _
class UnknownManagedObjectTypeError(CastellanException):
    message = _('Type not found, type: %(type)s')