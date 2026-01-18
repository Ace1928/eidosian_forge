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
def _find_network_id(cs, net_name):
    """Get unique network ID from network name from neutron"""
    try:
        return cs.neutron.find_network(net_name).id
    except (exceptions.NotFound, exceptions.NoUniqueMatch) as e:
        raise exceptions.CommandError(str(e))