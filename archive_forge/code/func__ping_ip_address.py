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
def _ping_ip_address(self, ip_address, should_succeed=True):
    cmd = ['ping', '-c1', '-w1', ip_address]

    def ping():
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.wait()
        return (proc.returncode == 0) == should_succeed
    return call_until_true(self.conf.build_timeout, 1, ping)