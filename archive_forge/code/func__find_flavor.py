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
def _find_flavor(cs, flavor):
    """Get a flavor by name, ID, or RAM size."""
    try:
        return utils.find_resource(cs.flavors, flavor, is_public=None)
    except exceptions.NotFound:
        return cs.flavors.find(ram=flavor)