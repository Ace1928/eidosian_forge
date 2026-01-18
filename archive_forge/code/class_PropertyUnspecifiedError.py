import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class PropertyUnspecifiedError(HeatException):
    msg_fmt = _('At least one of the following properties must be specified: %(props)s.')

    def __init__(self, *args, **kwargs):
        if args:
            kwargs.update({'props': ', '.join(args)})
        super(PropertyUnspecifiedError, self).__init__(**kwargs)