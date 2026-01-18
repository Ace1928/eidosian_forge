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
def _find_server(cs, server, raise_if_notfound=True, **find_args):
    """Get a server by name or ID.

    :param cs: NovaClient's instance
    :param server: identifier of server
    :param raise_if_notfound: raise an exception if server is not found
    :param find_args: argument to search server
    """
    if raise_if_notfound:
        return utils.find_resource(cs.servers, server, **find_args)
    else:
        try:
            return utils.find_resource(cs.servers, server, wrap_exception=False)
        except exceptions.NoUniqueMatch as e:
            raise exceptions.CommandError(str(e))
        except exceptions.NotFound:
            return server