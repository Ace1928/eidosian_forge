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
def _build_versioned_help_message(self, start_version, end_version):
    if start_version and end_version:
        msg = _(' (Supported by API versions %(start)s - %(end)s)') % {'start': start_version.get_string(), 'end': end_version.get_string()}
    elif start_version:
        msg = _(' (Supported by API version %(start)s and later)') % {'start': start_version.get_string()}
    else:
        msg = _(' (Supported until API version %(end)s)') % {'end': end_version.get_string()}
    return str(msg)