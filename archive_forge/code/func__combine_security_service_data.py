import ast
import re
import time
from oslo_utils import strutils
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import exceptions
from manilaclient.tests.functional import utils
def _combine_security_service_data(self, name=None, description=None, dns_ip=None, ou=None, server=None, domain=None, user=None, password=None, default_ad_site=None):
    data = ''
    if name is not None:
        data += '--name %s ' % name
    if description is not None:
        data += '--description %s ' % description
    if dns_ip is not None:
        data += '--dns-ip %s ' % dns_ip
    if ou is not None:
        data += '--ou %s ' % ou
    if server is not None:
        data += '--server %s ' % server
    if domain is not None:
        data += '--domain %s ' % domain
    if user is not None:
        data += '--user %s ' % user
    if password is not None:
        data += '--password %s ' % password
    if default_ad_site is not None:
        data += '--default-ad-site %s ' % default_ad_site
    return data