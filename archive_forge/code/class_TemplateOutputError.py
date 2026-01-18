import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class TemplateOutputError(HeatException):
    msg_fmt = _('Error in %(resource)s output %(attribute)s: %(message)s')