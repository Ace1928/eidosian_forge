import urllib
from castellan.i18n import _
class InsufficientCredentialDataError(CastellanException):
    message = _('Insufficient credential data was provided, either "token" must be set in the passed conf, or a context with an "auth_token" property must be passed.')