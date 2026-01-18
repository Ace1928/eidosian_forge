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
def _translate_extended_states(collection):
    power_states = ['NOSTATE', 'Running', '', 'Paused', 'Shutdown', '', 'Crashed', 'Suspended']
    for item in collection:
        try:
            setattr(item, 'power_state', power_states[getattr(item, 'power_state')])
        except AttributeError:
            setattr(item, 'power_state', 'N/A')
        try:
            getattr(item, 'task_state')
        except AttributeError:
            setattr(item, 'task_state', 'N/A')
        item.set_info('power_state', item.power_state)
        item.set_info('task_state', item.task_state)