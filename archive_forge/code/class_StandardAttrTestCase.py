import gc
from sqlalchemy.ext import declarative
from sqlalchemy import orm
import testtools
from neutron_lib.db import standard_attr
from neutron_lib.tests import _base as base
class StandardAttrTestCase(base.BaseTestCase):

    def setUp(self):
        super(StandardAttrTestCase, self).setUp()
        self.addCleanup(gc.collect)

    def _make_decl_base(self):
        try:

            class BaseV2(orm.DeclarativeBase, standard_attr.model_base.NeutronBaseV2):
                pass
            return BaseV2
        except AttributeError:
            return declarative.declarative_base(cls=standard_attr.model_base.NeutronBaseV2)

    def test_standard_attr_resource_model_map(self):
        rs_map = standard_attr.get_standard_attr_resource_model_map()
        base = self._make_decl_base()

        class MyModel(standard_attr.HasStandardAttributes, standard_attr.model_base.HasId, base):
            api_collections = ['my_resource', 'my_resource2']
            api_sub_resources = ['my_subresource']
        rs_map = standard_attr.get_standard_attr_resource_model_map()
        self.assertEqual(MyModel, rs_map['my_resource'])
        self.assertEqual(MyModel, rs_map['my_resource2'])
        self.assertEqual(MyModel, rs_map['my_subresource'])
        sub_rs_map = standard_attr.get_standard_attr_resource_model_map(include_resources=False, include_sub_resources=True)
        self.assertNotIn('my_resource', sub_rs_map)
        self.assertNotIn('my_resource2', sub_rs_map)
        self.assertEqual(MyModel, sub_rs_map['my_subresource'])
        nosub_rs_map = standard_attr.get_standard_attr_resource_model_map(include_resources=True, include_sub_resources=False)
        self.assertEqual(MyModel, nosub_rs_map['my_resource'])
        self.assertEqual(MyModel, nosub_rs_map['my_resource2'])
        self.assertNotIn('my_subresource', nosub_rs_map)

        class Dup(standard_attr.HasStandardAttributes, standard_attr.model_base.HasId, base):
            api_collections = ['my_resource']
        with testtools.ExpectedException(RuntimeError):
            standard_attr.get_standard_attr_resource_model_map()

    def test_standard_attr_resource_parent_map(self):
        base = self._make_decl_base()

        class TagSupportModel(standard_attr.HasStandardAttributes, standard_attr.model_base.HasId, base):
            collection_resource_map = {'collection_name': 'member_name'}
            tag_support = True

        class TagUnsupportModel(standard_attr.HasStandardAttributes, standard_attr.model_base.HasId, base):
            collection_resource_map = {'collection_name2': 'member_name2'}
            tag_support = False

        class TagUnsupportModel2(standard_attr.HasStandardAttributes, standard_attr.model_base.HasId, base):
            collection_resource_map = {'collection_name3': 'member_name3'}
        parent_map = standard_attr.get_tag_resource_parent_map()
        self.assertEqual('member_name', parent_map['collection_name'])
        self.assertNotIn('collection_name2', parent_map)
        self.assertNotIn('collection_name3', parent_map)

        class DupTagSupportModel(standard_attr.HasStandardAttributes, standard_attr.model_base.HasId, base):
            collection_resource_map = {'collection_name': 'member_name'}
            tag_support = True
        with testtools.ExpectedException(RuntimeError):
            standard_attr.get_tag_resource_parent_map()