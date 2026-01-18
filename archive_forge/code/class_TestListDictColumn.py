import collections
from osc_lib.cli import format_columns
from osc_lib.tests import utils
class TestListDictColumn(utils.TestCase):

    def test_list_dict_column(self):
        data = [{'key1': 'value1'}, {'key2': 'value2'}]
        col = format_columns.ListDictColumn(data)
        self.assertEqual(data, col.machine_readable())
        self.assertEqual("key1='value1'\nkey2='value2'", col.human_readable())

    def test_complex_object(self):
        """Non-list-of-dict objects should be converted to a list-of-dicts."""
        data = (collections.OrderedDict([('key1', 'value1'), ('key2', 'value2')]),)
        col = format_columns.ListDictColumn(data)
        self.assertEqual(type(col.machine_readable()), list)
        for x in col.machine_readable():
            self.assertEqual(type(x), dict)