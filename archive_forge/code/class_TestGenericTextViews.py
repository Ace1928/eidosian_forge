import copy
from unittest import mock
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports.models import with_default_views as mwdv
from oslo_reports import report
from oslo_reports.views import jinja_view as jv
from oslo_reports.views.json import generic as json_generic
from oslo_reports.views.text import generic as text_generic
class TestGenericTextViews(base.BaseTestCase):

    def setUp(self):
        super(TestGenericTextViews, self).setUp()
        self.model = mwdv_generator()
        self.model.set_current_view_type('text')

    def test_multi_view(self):
        attached_view = text_generic.MultiView()
        self.model = base_model.ReportModel(data={}, attached_view=attached_view)
        self.model['1'] = mwdv_generator()
        self.model['2'] = mwdv_generator()
        self.model['2']['int'] = 2
        self.model.set_current_view_type('text')
        target_str = 'int = 1\nstring = value\nint = 2\nstring = value'
        self.assertEqual(target_str, str(self.model))

    def test_basic_kv_view(self):
        attached_view = text_generic.BasicKeyValueView()
        self.model = base_model.ReportModel(data={'string': 'value', 'int': 1}, attached_view=attached_view)
        self.assertEqual('int = 1\nstring = value\n', str(self.model))

    def test_table_view(self):
        column_names = ['Column A', 'Column B']
        column_values = ['a', 'b']
        attached_view = text_generic.TableView(column_names, column_values, 'table')
        self.model = base_model.ReportModel(data={}, attached_view=attached_view)
        self.model['table'] = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
        target_str = '             Column A              |             Column B               \n------------------------------------------------------------------------\n                 1                 |                 2                  \n                 3                 |                 4                  \n'
        self.assertEqual(target_str, str(self.model))

    def test_dict_serialization(self):
        self.model['dt'] = {'a': 1, 'b': 2}
        target_str = 'dt = \n  a = 1\n  b = 2\nint = 1\nstring = value'
        self.assertEqual(target_str, str(self.model))

    def test_dict_serialization_integer_keys(self):
        self.model['dt'] = {3: 4, 5: 6}
        target_str = 'dt = \n  3 = 4\n  5 = 6\nint = 1\nstring = value'
        self.assertEqual(target_str, str(self.model))

    def test_dict_serialization_mixed_keys(self):
        self.model['dt'] = {'3': 4, 5: 6}
        target_str = 'dt = \n  3 = 4\n  5 = 6\nint = 1\nstring = value'
        self.assertEqual(target_str, str(self.model))

    def test_list_serialization(self):
        self.model['lt'] = ['a', 'b']
        target_str = 'int = 1\nlt = \n  a\n  b\nstring = value'
        self.assertEqual(target_str, str(self.model))

    def test_list_in_dict_serialization(self):
        self.model['dt'] = {'a': 1, 'b': [2, 3]}
        target_str = 'dt = \n  a = 1\n  b = \n    2\n    3\nint = 1\nstring = value'
        self.assertEqual(target_str, str(self.model))

    def test_dict_in_list_serialization(self):
        self.model['lt'] = [1, {'b': 2, 'c': 3}]
        target_str = 'int = 1\nlt = \n  1\n  [dict]\n    b = 2\n    c = 3\nstring = value'
        self.assertEqual(target_str, str(self.model))

    def test_submodel_serialization(self):
        sm = mwdv_generator()
        sm.set_current_view_type('text')
        self.model['submodel'] = sm
        target_str = 'int = 1\nstring = value\nsubmodel = \n  int = 1\n  string = value'
        self.assertEqual(target_str, str(self.model))

    def test_custom_indent_string(self):
        view = text_generic.KeyValueView(indent_str='~~')
        self.model['lt'] = ['a', 'b']
        self.model.attached_view = view
        target_str = 'int = 1\nlt = \n~~a\n~~b\nstring = value'
        self.assertEqual(target_str, str(self.model))