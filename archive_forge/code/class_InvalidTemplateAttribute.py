import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class InvalidTemplateAttribute(HeatException):
    msg_fmt = _('The Referenced Attribute (%(resource)s %(key)s) is incorrect.')