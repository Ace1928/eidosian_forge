import os
import random
import re
import subprocess
import time
import urllib
import fixtures
from heatclient import exc as heat_exceptions
from keystoneauth1 import exceptions as kc_exceptions
from oslo_log import log as logging
from oslo_utils import timeutils
from tempest import config
import testscenarios
import testtools
from heat_integrationtests.common import clients
from heat_integrationtests.common import exceptions
def _wait_for_stack_status(self, stack_identifier, status, failure_pattern=None, success_on_not_found=False, signal_required=False, resources_to_signal=None, is_action_cancelled=False):
    """Waits for a Stack to reach a given status.

        Note this compares the full $action_$status, e.g
        CREATE_COMPLETE, not just COMPLETE which is exposed
        via the status property of Stack in heatclient
        """
    if failure_pattern:
        fail_regexp = re.compile(failure_pattern)
    elif 'FAILED' in status:
        fail_regexp = re.compile('^.*_COMPLETE$')
    else:
        fail_regexp = re.compile('^.*_FAILED$')
    build_timeout = self.conf.build_timeout
    build_interval = self.conf.build_interval
    start = timeutils.utcnow()
    while timeutils.delta_seconds(start, timeutils.utcnow()) < build_timeout:
        try:
            stack = self.client.stacks.get(stack_identifier, resolve_outputs=False)
        except heat_exceptions.HTTPNotFound:
            if success_on_not_found:
                return
            elif not any((s in status for s in ['CREATE', 'ADOPT'])):
                raise
        else:
            if self._verify_status(stack, stack_identifier, status, fail_regexp, is_action_cancelled):
                return
        if signal_required:
            self.signal_resources(resources_to_signal)
        time.sleep(build_interval)
    message = 'Stack %s failed to reach %s status within the required time (%s s). Current stack state: %s.' % (stack_identifier, status, build_timeout, stack.stack_status)
    raise exceptions.TimeoutException(message)