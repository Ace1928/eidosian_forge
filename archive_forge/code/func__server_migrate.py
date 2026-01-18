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
def _server_migrate(cs, server):
    success = True
    error_message = ''
    try:
        cs.servers.migrate(server['uuid'])
    except Exception as e:
        success = False
        error_message = _('Error while migrating instance: %s') % e
    return HostServersMigrateResponse(base.Manager, {'server_uuid': server['uuid'], 'migration_accepted': success, 'error_message': error_message})