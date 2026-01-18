import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class InvalidTenant(HeatException):
    msg_fmt = _('Searching Tenant %(target)s from Tenant %(actual)s forbidden.')