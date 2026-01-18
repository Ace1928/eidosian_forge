from blazarclient.i18n import _
class ResourcePropertyNotFound(BlazarClientException):
    """Occurs if the resource property specified does not exist"""
    message = _('The resource property does not exist.')
    code = 404