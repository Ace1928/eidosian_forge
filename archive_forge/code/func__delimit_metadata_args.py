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
def _delimit_metadata_args(self, argv):
    """This function adds -- separator at the appropriate spot
        """
    word = '--metadata'
    tmp = []
    metadata_options = False
    if word in argv:
        for arg in argv:
            if arg == word:
                metadata_options = True
            elif metadata_options:
                if arg.startswith('--'):
                    metadata_options = False
                elif '=' not in arg:
                    tmp.append(u'--')
                    metadata_options = False
            tmp.append(arg)
        return tmp
    else:
        return argv