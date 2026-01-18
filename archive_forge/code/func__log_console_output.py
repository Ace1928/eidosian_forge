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
def _log_console_output(self, servers=None):
    if not servers:
        servers = self.compute_client.servers.list()
    for server in servers:
        LOG.info('Console output for %s', server.id)
        LOG.info(server.get_console_output())