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
def _server_evacuate(cs, server, args):
    success = True
    error_message = ''
    try:
        if api_versions.APIVersion('2.68') <= cs.api_version:
            cs.servers.evacuate(server=server['uuid'], host=args.target_host)
        elif api_versions.APIVersion('2.29') <= cs.api_version:
            force = getattr(args, 'force', None)
            cs.servers.evacuate(server=server['uuid'], host=args.target_host, force=force)
        elif api_versions.APIVersion('2.14') <= cs.api_version:
            cs.servers.evacuate(server=server['uuid'], host=args.target_host)
        else:
            on_shared_storage = getattr(args, 'on_shared_storage', None)
            cs.servers.evacuate(server=server['uuid'], host=args.target_host, on_shared_storage=on_shared_storage)
    except Exception as e:
        success = False
        error_message = _('Error while evacuating instance: %s') % e
    return EvacuateHostResponse(base.Manager, {'server_uuid': server['uuid'], 'evacuate_accepted': success, 'error_message': error_message})