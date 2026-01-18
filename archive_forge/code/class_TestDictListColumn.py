import collections
from osc_lib.cli import format_columns
from osc_lib.tests import utils
class TestDictListColumn(utils.TestCase):

    def test_dict_list_column(self):
        data = {'public': ['2001:db8::8', '172.24.4.6'], 'private': ['2000:db7::7', '192.24.4.6']}
        col = format_columns.DictListColumn(data)
        self.assertEqual(data, col.machine_readable())
        self.assertEqual('private=192.24.4.6, 2000:db7::7; public=172.24.4.6, 2001:db8::8', col.human_readable())

    def test_complex_object(self):
        """Non-dict-of-list objects should be converted to a dict-of-lists."""
        data = collections.OrderedDict([('key1', ['value1']), ('key2', ['value2'])])
        col = format_columns.DictListColumn(data)
        self.assertEqual(type(col.machine_readable()), dict)