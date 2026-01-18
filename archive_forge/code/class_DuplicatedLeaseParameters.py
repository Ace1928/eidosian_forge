from blazarclient.i18n import _
class DuplicatedLeaseParameters(BlazarClientException):
    """Occurs if lease parameters are duplicated."""
    message = _('The lease parameters are duplicated.')
    code = 400