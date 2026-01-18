import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class InvalidTemplateVersion(HeatException):
    msg_fmt = _('The template version is invalid: %(explanation)s')