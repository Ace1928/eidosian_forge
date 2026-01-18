import logging
from oslo_concurrency import lockutils
from oslo_context import context
from oslo_utils import excutils
from oslo_utils import reflection
from oslo_vmware._i18n import _
from oslo_vmware.common import loopingcall
from oslo_vmware import exceptions
from oslo_vmware import pbm
from oslo_vmware import vim
from oslo_vmware import vim_util
def _trunc_id(session_id):
    """Returns truncated session id which is suitable for logging."""
    if session_id is not None:
        return session_id[-5:]