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
def _server_live_migrate(cs, server, args):

    class HostEvacuateLiveResponse(object):

        def __init__(self, server_uuid, live_migration_accepted, error_message):
            self.server_uuid = server_uuid
            self.live_migration_accepted = live_migration_accepted
            self.error_message = error_message
    success = True
    error_message = ''
    update_kwargs = {}
    try:
        if 'force' in args and args.force:
            update_kwargs['force'] = args.force
        if 'disk_over_commit' in args:
            update_kwargs['disk_over_commit'] = args.disk_over_commit
        cs.servers.live_migrate(server['uuid'], args.target_host, args.block_migrate, **update_kwargs)
    except Exception as e:
        success = False
        error_message = _('Error while live migrating instance: %s') % e
    return HostEvacuateLiveResponse(server['uuid'], success, error_message)