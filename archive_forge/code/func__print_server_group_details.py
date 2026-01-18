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
def _print_server_group_details(cs, server_group):
    if cs.api_version < api_versions.APIVersion('2.13'):
        columns = ['Id', 'Name', 'Policies', 'Members', 'Metadata']
    elif cs.api_version < api_versions.APIVersion('2.64'):
        columns = ['Id', 'Name', 'Project Id', 'User Id', 'Policies', 'Members', 'Metadata']
    else:
        columns = ['Id', 'Name', 'Project Id', 'User Id', 'Policy', 'Rules', 'Members']
    utils.print_list(server_group, columns)