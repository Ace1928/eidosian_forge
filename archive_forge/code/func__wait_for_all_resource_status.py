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
def _wait_for_all_resource_status(self, stack_identifier, status, failure_pattern='^.*_FAILED$', success_on_not_found=False):
    for res in self.client.resources.list(stack_identifier):
        self._wait_for_resource_status(stack_identifier, res.resource_name, status, failure_pattern=failure_pattern, success_on_not_found=success_on_not_found)