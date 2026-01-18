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
def _poll_for_status(poll_fn, obj_id, action, final_ok_states, poll_period=5, show_progress=True, status_field='status', silent=False):
    """Block while an action is being performed, periodically printing
    progress.
    """

    def print_progress(progress):
        if show_progress:
            msg = _('\rServer %(action)s... %(progress)s%% complete') % dict(action=action, progress=progress)
        else:
            msg = _('\rServer %(action)s...') % dict(action=action)
        sys.stdout.write(msg)
        sys.stdout.flush()
    if not silent:
        print()
    while True:
        obj = poll_fn(obj_id)
        status = getattr(obj, status_field)
        if status:
            status = status.lower()
        progress = getattr(obj, 'progress', None) or 0
        if status in final_ok_states:
            if not silent:
                print_progress(100)
                print(_('\nFinished'))
            break
        elif status == 'error':
            if not silent:
                print(_('\nError %s server') % action)
            raise exceptions.ResourceInErrorState(obj)
        elif status == 'deleted':
            if not silent:
                print(_('\nDeleted %s server') % action)
            raise exceptions.InstanceInDeletedState(obj.fault['message'])
        if not silent:
            print_progress(progress)
        time.sleep(poll_period)