import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def _test_get_dict_properties_with_formatter(self, formatters):
    names = ('id', 'attr')
    item = {'id': 'fake-id', 'attr': ['a', 'b']}
    res_id, res_attr = utils.get_dict_properties(item, names, formatters=formatters)
    self.assertEqual('fake-id', res_id)
    return res_attr