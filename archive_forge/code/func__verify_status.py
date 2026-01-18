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
def _verify_status(self, stack, stack_identifier, status, fail_regexp, is_action_cancelled=False):
    if stack.stack_status == status:
        if status == 'DELETE_COMPLETE' and stack.deletion_time is None:
            return False
        else:
            return True
    wait_for_action = status.split('_')[0]
    if stack.action == wait_for_action and fail_regexp.search(stack.stack_status):
        raise exceptions.StackBuildErrorException(stack_identifier=stack_identifier, stack_status=stack.stack_status, stack_status_reason=stack.stack_status_reason)
    return False