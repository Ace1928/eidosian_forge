import functools
import re
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import units
import six
from os_win._i18n import _
from os_win import conf
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
def _create_security_acl(self, sg_rule, weight):
    acl = super(NetworkUtilsR2, self)._create_security_acl(sg_rule, weight)
    acl.Weight = weight
    sg_rule.Weight = weight
    return acl