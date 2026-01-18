import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class InvalidEncryptionKey(HeatException):
    msg_fmt = _('Can not decrypt data with the auth_encryption_key in heat config.')