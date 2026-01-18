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
class FormattedInterface(object):

    def __init__(self, interface):
        for col in columns:
            key = col.lower().replace(' ', '_')
            if hasattr(interface, key):
                setattr(self, key, getattr(interface, key))
        self.ip_addresses = ','.join([fip['ip_address'] for fip in interface.fixed_ips])