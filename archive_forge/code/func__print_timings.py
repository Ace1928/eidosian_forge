import argparse
import collections
import getpass
import logging
import sys
from urllib import parse as urlparse
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1.identity import v2 as v2_auth
from keystoneauth1.identity import v3 as v3_auth
from keystoneauth1 import loading
from keystoneauth1 import session
from oslo_utils import importutils
import requests
import cinderclient
from cinderclient._i18n import _
from cinderclient import api_versions
from cinderclient import client
from cinderclient import exceptions as exc
from cinderclient import utils
def _print_timings(self, session):
    timings = session.get_timings()
    utils.print_list(timings, fields=('method', 'url', 'seconds'), sortby_index=None, formatters={'seconds': lambda r: r.elapsed.total_seconds()})