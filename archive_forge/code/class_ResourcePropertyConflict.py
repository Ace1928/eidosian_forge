import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class ResourcePropertyConflict(HeatException):
    msg_fmt = _('Cannot define the following properties at the same time: %(props)s.')

    def __init__(self, *args, **kwargs):
        if args:
            kwargs.update({'props': ', '.join(args)})
        super(ResourcePropertyConflict, self).__init__(**kwargs)