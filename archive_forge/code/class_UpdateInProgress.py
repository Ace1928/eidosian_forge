import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class UpdateInProgress(Exception):

    def __init__(self, resource_name='Unknown'):
        msg = _('The resource %s is already being updated.') % resource_name
        super(Exception, self).__init__(str(msg))