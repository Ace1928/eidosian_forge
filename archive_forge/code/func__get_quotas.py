from random import randint
import ddt
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient import api_versions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def _get_quotas(self, role, operation, microversion):
    roles = self.parser.listing(self.clients[role].manila('quota-%s' % operation, microversion=microversion))
    self.assertTableStruct(roles, ('Property', 'Value'))