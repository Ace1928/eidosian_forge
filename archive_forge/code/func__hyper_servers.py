import argparse
import collections
import datetime
import getpass
import logging
import os
import pprint
import sys
import time
from oslo_utils import netutils
from oslo_utils import strutils
from oslo_utils import timeutils
import novaclient
from novaclient import api_versions
from novaclient import base
from novaclient import client
from novaclient import exceptions
from novaclient.i18n import _
from novaclient import shell
from novaclient import utils
from novaclient.v2 import availability_zones
from novaclient.v2 import quotas
from novaclient.v2 import servers
def _hyper_servers(cs, host, strict):
    hypervisors = cs.hypervisors.search(host, servers=True)
    for hyper in hypervisors:
        if strict and hyper.hypervisor_hostname != host:
            continue
        if hasattr(hyper, 'servers'):
            for server in hyper.servers:
                yield server
        if strict:
            break
    else:
        if strict:
            msg = _("No hypervisor matching '%s' could be found.") % host
            raise exceptions.NotFound(404, msg)