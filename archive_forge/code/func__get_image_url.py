import argparse
import copy
import getpass
import hashlib
import json
import logging
import os
import sys
import traceback
from oslo_utils import encodeutils
from oslo_utils import importutils
import urllib.parse
import glanceclient
from glanceclient._i18n import _
from glanceclient.common import utils
from glanceclient import exc
from keystoneauth1 import discover
from keystoneauth1 import exceptions as ks_exc
from keystoneauth1.identity import v2 as v2_auth
from keystoneauth1.identity import v3 as v3_auth
from keystoneauth1 import loading
def _get_image_url(self, args):
    """Translate the available url-related options into a single string.

        Return the endpoint that should be used to talk to Glance if a
        clear decision can be made. Otherwise, return None.
        """
    if args.os_image_url:
        return args.os_image_url
    else:
        return None