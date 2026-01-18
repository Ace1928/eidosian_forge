from blazarclient.i18n import _
class NoBlazarEndpoint(BlazarClientException):
    """Occurs if no endpoint for Blazar set in the Keystone."""
    message = _('No publicURL endpoint for Blazar found. Set endpoint for Blazar in the Keystone.')
    code = 404