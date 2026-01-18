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
@not_found_wrapper
def access_allow(self, share_id, access_type, access_to, access_level, metadata=None, microversion=None):
    cmd = 'access-allow  --access-level %(level)s %(id)s %(type)s %(access_to)s' % {'level': access_level, 'id': share_id, 'type': access_type, 'access_to': access_to}
    if metadata:
        metadata_cli = ''
        for k, v in metadata.items():
            metadata_cli += '%(k)s=%(v)s ' % {'k': k, 'v': v}
        if metadata_cli:
            cmd += ' --metadata %s ' % metadata_cli
    raw_access = self.manila(cmd, microversion=microversion)
    return output_parser.details(raw_access)