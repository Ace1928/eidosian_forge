from blazarclient.i18n import _
class InsufficientAuthInformation(BlazarClientException):
    """Occurs if the auth info passed to blazar client is insufficient."""
    message = _('The passed arguments are insufficient for the authentication. The instance of keystoneauth1.session.Session class is required.')
    code = 400