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
def _stack_delete(self, stack_identifier):
    try:
        self._handle_in_progress(self.client.stacks.delete, stack_identifier)
    except heat_exceptions.HTTPNotFound:
        pass
    self._wait_for_stack_status(stack_identifier, 'DELETE_COMPLETE', success_on_not_found=True)