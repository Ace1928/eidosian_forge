import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class ResourcePropertyValueDependency(HeatException):
    msg_fmt = _('%(prop1)s property should only be specified for %(prop2)s with value %(value)s.')