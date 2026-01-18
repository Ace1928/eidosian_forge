import urllib
from castellan.i18n import _
class InvalidManagedObjectDictError(CastellanException):
    message = _("Dict has no field '%(field)s'.")