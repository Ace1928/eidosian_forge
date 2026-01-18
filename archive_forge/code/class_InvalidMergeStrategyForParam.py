import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class InvalidMergeStrategyForParam(HeatException):
    msg_fmt = _("Invalid merge strategy '%(strategy)s' for parameter '%(param)s'.")