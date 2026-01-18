from concurrent import futures
import enum
import errno
import io
import logging as pylogging
import os
import platform
import socket
import subprocess
import sys
import tempfile
import threading
import eventlet
from eventlet import patcher
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_privsep._i18n import _
from oslo_privsep import capabilities
from oslo_privsep import comm
def _drop_privs(self):
    try:
        capabilities.set_keepcaps(True)
        if self.group is not None:
            try:
                os.setgroups([])
            except OSError:
                msg = _('Failed to remove supplemental groups')
                LOG.critical(msg)
                raise FailedToDropPrivileges(msg)
            setgid(self.group)
        if self.user is not None:
            setuid(self.user)
    finally:
        capabilities.set_keepcaps(False)
    LOG.info('privsep process running with uid/gid: %(uid)s/%(gid)s', {'uid': os.getuid(), 'gid': os.getgid()})
    capabilities.drop_all_caps_except(self.caps, self.caps, [])

    def fmt_caps(capset):
        if not capset:
            return 'none'
        fc = [capabilities.CAPS_BYVALUE.get(c, str(c)) for c in capset]
        fc.sort()
        return '|'.join(fc)
    eff, prm, inh = capabilities.get_caps()
    LOG.info('privsep process running with capabilities (eff/prm/inh): %(eff)s/%(prm)s/%(inh)s', {'eff': fmt_caps(eff), 'prm': fmt_caps(prm), 'inh': fmt_caps(inh)})