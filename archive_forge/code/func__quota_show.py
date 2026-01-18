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
def _quota_show(quotas):

    class FormattedQuota(object):

        def __init__(self, key, value):
            setattr(self, 'quota', key)
            setattr(self, 'limit', value)
    quota_list = []
    for resource in _quota_resources:
        try:
            quota = FormattedQuota(resource, getattr(quotas, resource))
            quota_list.append(quota)
        except AttributeError:
            pass
    columns = ['Quota', 'Limit']
    utils.print_list(quota_list, columns)