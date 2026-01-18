import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class InvalidTemplateVersions(HeatException):
    msg_fmt = _('A template version alias %(version)s was added for a template class that has no official YYYY-MM-DD version.')