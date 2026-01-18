from oslo_utils import excutils
from neutron_lib._i18n import _
class NetworkTunnelRangeError(NeutronException):
    """An error due to an invalid network tunnel range.

    An exception indicating an invalid network tunnel range was specified.

    :param tunnel_range: The invalid tunnel range. If specified in the
        start:end' format, they will be converted automatically.
    :param error: Additional details on why the range is invalid.
    """
    message = _("Invalid network tunnel range: '%(tunnel_range)s' - %(error)s.")

    def __init__(self, **kwargs):
        if isinstance(kwargs['tunnel_range'], tuple):
            kwargs['tunnel_range'] = '%d:%d' % kwargs['tunnel_range']
        super().__init__(**kwargs)