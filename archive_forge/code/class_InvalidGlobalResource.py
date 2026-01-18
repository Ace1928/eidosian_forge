import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class InvalidGlobalResource(HeatException):
    msg_fmt = _('There was an error loading the definition of the global resource type %(type_name)s.')