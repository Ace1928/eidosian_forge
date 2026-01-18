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
def _meta_parsing(metadata):
    try:
        return dict((v.split('=', 1) for v in metadata))
    except ValueError:
        msg = _("'%s' is not in the format of 'key=value'") % metadata
        raise argparse.ArgumentTypeError(msg)