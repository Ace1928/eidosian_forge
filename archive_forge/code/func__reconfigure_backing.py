import os
import tempfile
from oslo_log import log as logging
from oslo_utils import fileutils
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator import initiator_connector
def _reconfigure_backing(self, session, backing, reconfig_spec):
    LOG.debug('Reconfiguring backing VM: %(backing)s with spec: %(spec)s.', {'backing': backing, 'spec': reconfig_spec})
    reconfig_task = session.invoke_api(session.vim, 'ReconfigVM_Task', backing, spec=reconfig_spec)
    LOG.debug('Task: %s created for reconfiguring backing VM.', reconfig_task)
    session.wait_for_task(reconfig_task)