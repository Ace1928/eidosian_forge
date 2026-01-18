import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
class MetadefTagTests(object):

    def test_tag_create(self):
        fixture = build_namespace_fixture()
        created_ns = self.db_api.metadef_namespace_create(self.context, fixture)
        self.assertIsNotNone(created_ns)
        self._assert_saved_fields(fixture, created_ns)
        fixture_tag = build_tag_fixture(namespace_id=created_ns['id'])
        created_tag = self.db_api.metadef_tag_create(self.context, created_ns['namespace'], fixture_tag)
        self._assert_saved_fields(fixture_tag, created_tag)

    def test_tag_create_duplicate(self):
        fixture = build_namespace_fixture()
        created_ns = self.db_api.metadef_namespace_create(self.context, fixture)
        self.assertIsNotNone(created_ns)
        self._assert_saved_fields(fixture, created_ns)
        fixture_tag = build_tag_fixture(namespace_id=created_ns['id'])
        created_tag = self.db_api.metadef_tag_create(self.context, created_ns['namespace'], fixture_tag)
        self._assert_saved_fields(fixture_tag, created_tag)
        self.assertRaises(exception.Duplicate, self.db_api.metadef_tag_create, self.context, created_ns['namespace'], fixture_tag)

    def test_tag_create_tags(self):
        fixture = build_namespace_fixture()
        created_ns = self.db_api.metadef_namespace_create(self.context, fixture)
        self.assertIsNotNone(created_ns)
        self._assert_saved_fields(fixture, created_ns)
        tags = build_tags_fixture(['Tag1', 'Tag2', 'Tag3'])
        created_tags = self.db_api.metadef_tag_create_tags(self.context, created_ns['namespace'], tags)
        actual = set([tag['name'] for tag in created_tags])
        expected = set(['Tag1', 'Tag2', 'Tag3'])
        self.assertEqual(expected, actual)

    def test_tag_create_tags_with_append(self):
        fixture = build_namespace_fixture()
        created_ns = self.db_api.metadef_namespace_create(self.context, fixture)
        self.assertIsNotNone(created_ns)
        self._assert_saved_fields(fixture, created_ns)
        tags = build_tags_fixture(['Tag1', 'Tag2', 'Tag3'])
        created_tags = self.db_api.metadef_tag_create_tags(self.context, created_ns['namespace'], tags)
        actual = set([tag['name'] for tag in created_tags])
        expected = set(['Tag1', 'Tag2', 'Tag3'])
        self.assertEqual(expected, actual)
        new_tags = build_tags_fixture(['Tag4', 'Tag5', 'Tag6'])
        new_created_tags = self.db_api.metadef_tag_create_tags(self.context, created_ns['namespace'], new_tags, can_append=True)
        actual = set([tag['name'] for tag in new_created_tags])
        expected = set(['Tag4', 'Tag5', 'Tag6'])
        self.assertEqual(expected, actual)
        tags = self.db_api.metadef_tag_get_all(self.context, created_ns['namespace'], sort_key='created_at')
        actual = set([tag['name'] for tag in tags])
        expected = set(['Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5', 'Tag6'])
        self.assertEqual(expected, actual)

    def test_tag_create_duplicate_tags_1(self):
        fixture = build_namespace_fixture()
        created_ns = self.db_api.metadef_namespace_create(self.context, fixture)
        self.assertIsNotNone(created_ns)
        self._assert_saved_fields(fixture, created_ns)
        tags = build_tags_fixture(['Tag1', 'Tag2', 'Tag3', 'Tag2'])
        self.assertRaises(exception.Duplicate, self.db_api.metadef_tag_create_tags, self.context, created_ns['namespace'], tags)

    def test_tag_create_duplicate_tags_2(self):
        fixture = build_namespace_fixture()
        created_ns = self.db_api.metadef_namespace_create(self.context, fixture)
        self.assertIsNotNone(created_ns)
        self._assert_saved_fields(fixture, created_ns)
        tags = build_tags_fixture(['Tag1', 'Tag2', 'Tag3'])
        self.db_api.metadef_tag_create_tags(self.context, created_ns['namespace'], tags)
        dup_tag = build_tag_fixture(namespace_id=created_ns['id'], name='Tag3')
        self.assertRaises(exception.Duplicate, self.db_api.metadef_tag_create, self.context, created_ns['namespace'], dup_tag)

    def test_tag_create_duplicate_tags_3(self):
        fixture = build_namespace_fixture()
        created_ns = self.db_api.metadef_namespace_create(self.context, fixture)
        self.assertIsNotNone(created_ns)
        self._assert_saved_fields(fixture, created_ns)
        tags = build_tags_fixture(['Tag1', 'Tag2', 'Tag3'])
        self.db_api.metadef_tag_create_tags(self.context, created_ns['namespace'], tags)
        dup_tags = build_tags_fixture(['Tag3', 'Tag4', 'Tag5'])
        self.assertRaises(exception.Duplicate, self.db_api.metadef_tag_create_tags, self.context, created_ns['namespace'], dup_tags, can_append=True)

    def test_tag_get(self):
        fixture_ns = build_namespace_fixture()
        created_ns = self.db_api.metadef_namespace_create(self.context, fixture_ns)
        self.assertIsNotNone(created_ns)
        self._assert_saved_fields(fixture_ns, created_ns)
        fixture_tag = build_tag_fixture(namespace_id=created_ns['id'])
        created_tag = self.db_api.metadef_tag_create(self.context, created_ns['namespace'], fixture_tag)
        found_tag = self.db_api.metadef_tag_get(self.context, created_ns['namespace'], created_tag['name'])
        self._assert_saved_fields(fixture_tag, found_tag)

    def test_tag_get_all(self):
        ns_fixture = build_namespace_fixture()
        ns_created = self.db_api.metadef_namespace_create(self.context, ns_fixture)
        self.assertIsNotNone(ns_created, 'Could not create a namespace.')
        self._assert_saved_fields(ns_fixture, ns_created)
        fixture1 = build_tag_fixture(namespace_id=ns_created['id'])
        created_tag1 = self.db_api.metadef_tag_create(self.context, ns_created['namespace'], fixture1)
        self.assertIsNotNone(created_tag1, 'Could not create tag 1.')
        fixture2 = build_tag_fixture(namespace_id=ns_created['id'], name='test-tag-2')
        created_tag2 = self.db_api.metadef_tag_create(self.context, ns_created['namespace'], fixture2)
        self.assertIsNotNone(created_tag2, 'Could not create tag 2.')
        found = self.db_api.metadef_tag_get_all(self.context, ns_created['namespace'], sort_key='created_at')
        self.assertEqual(2, len(found))

    def test_tag_update(self):
        delta = {'name': 'New-name'}
        fixture_ns = build_namespace_fixture()
        created_ns = self.db_api.metadef_namespace_create(self.context, fixture_ns)
        self.assertIsNotNone(created_ns['namespace'])
        tag_fixture = build_tag_fixture(namespace_id=created_ns['id'])
        created_tag = self.db_api.metadef_tag_create(self.context, created_ns['namespace'], tag_fixture)
        self.assertIsNotNone(created_tag, 'Could not create a tag.')
        delta_dict = {}
        delta_dict.update(delta.copy())
        updated = self.db_api.metadef_tag_update(self.context, created_ns['namespace'], created_tag['id'], delta_dict)
        self.assertEqual(delta['name'], updated['name'])

    def test_tag_delete(self):
        fixture_ns = build_namespace_fixture()
        created_ns = self.db_api.metadef_namespace_create(self.context, fixture_ns)
        self.assertIsNotNone(created_ns['namespace'])
        tag_fixture = build_tag_fixture(namespace_id=created_ns['id'])
        created_tag = self.db_api.metadef_tag_create(self.context, created_ns['namespace'], tag_fixture)
        self.assertIsNotNone(created_tag, 'Could not create a tag.')
        self.db_api.metadef_tag_delete(self.context, created_ns['namespace'], created_tag['name'])
        self.assertRaises(exception.NotFound, self.db_api.metadef_tag_get, self.context, created_ns['namespace'], created_tag['name'])