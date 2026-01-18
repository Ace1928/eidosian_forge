import copy
from unittest import mock
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports.models import with_default_views as mwdv
from oslo_reports import report
from oslo_reports.views import jinja_view as jv
from oslo_reports.views.json import generic as json_generic
from oslo_reports.views.text import generic as text_generic
class TestGenericJSONViews(base.BaseTestCase):

    def setUp(self):
        super(TestGenericJSONViews, self).setUp()
        self.model = mwdv_generator()
        self.model.set_current_view_type('json')

    def test_basic_kv_view(self):
        attached_view = json_generic.BasicKeyValueView()
        self.model = base_model.ReportModel(data={'string': 'value', 'int': 1}, attached_view=attached_view)
        self.assertEqual('{"int": 1, "string": "value"}', str(self.model))

    def test_dict_serialization(self):
        self.model['dt'] = {'a': 1, 'b': 2}
        target_str = '{"dt": {"a": 1, "b": 2}, "int": 1, "string": "value"}'
        self.assertEqual(target_str, str(self.model))

    def test_list_serialization(self):
        self.model['lt'] = ['a', 'b']
        target_str = '{"int": 1, "lt": ["a", "b"], "string": "value"}'
        self.assertEqual(target_str, str(self.model))

    def test_list_in_dict_serialization(self):
        self.model['dt'] = {'a': 1, 'b': [2, 3]}
        target_str = '{"dt": {"a": 1, "b": [2, 3]}, "int": 1, "string": "value"}'
        self.assertEqual(target_str, str(self.model))

    def test_dict_in_list_serialization(self):
        self.model['lt'] = [1, {'b': 2, 'c': 3}]
        target_str = '{"int": 1, "lt": [1, {"b": 2, "c": 3}], "string": "value"}'
        self.assertEqual(target_str, str(self.model))

    def test_submodel_serialization(self):
        sm = mwdv_generator()
        sm.set_current_view_type('json')
        self.model['submodel'] = sm
        target_str = '{"int": 1, "string": "value", "submodel": {"int": 1, "string": "value"}}'
        self.assertEqual(target_str, str(self.model))