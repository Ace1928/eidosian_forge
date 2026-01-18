import json
import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
@staticmethod
def _get_random_trait():
    return data_utils.rand_name('CUSTOM', '').replace('-', '_')