import copy
import datetime
import functools
from unittest import mock
import uuid
from oslo_db import exception as db_exception
from oslo_db.sqlalchemy import utils as sqlalchemyutils
from sqlalchemy import sql
from glance.common import exception
from glance.common import timeutils
from glance import context
from glance.db.sqlalchemy import api as db_api
from glance.db.sqlalchemy import models
from glance.tests import functional
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
class DriverTests(object):

    def test_image_create_requires_status(self):
        fixture = {'name': 'mark', 'size': 12}
        self.assertRaises(exception.Invalid, self.db_api.image_create, self.context, fixture)
        fixture = {'name': 'mark', 'size': 12, 'status': 'queued'}
        self.db_api.image_create(self.context, fixture)

    @mock.patch.object(timeutils, 'utcnow')
    def test_image_create_defaults(self, mock_utcnow):
        mock_utcnow.return_value = datetime.datetime.utcnow()
        create_time = timeutils.utcnow()
        values = {'status': 'queued', 'created_at': create_time, 'updated_at': create_time}
        image = self.db_api.image_create(self.context, values)
        self.assertIsNone(image['name'])
        self.assertIsNone(image['container_format'])
        self.assertEqual(0, image['min_ram'])
        self.assertEqual(0, image['min_disk'])
        self.assertIsNone(image['owner'])
        self.assertEqual('shared', image['visibility'])
        self.assertIsNone(image['size'])
        self.assertIsNone(image['checksum'])
        self.assertIsNone(image['disk_format'])
        self.assertEqual([], image['locations'])
        self.assertFalse(image['protected'])
        self.assertFalse(image['deleted'])
        self.assertIsNone(image['deleted_at'])
        self.assertEqual([], image['properties'])
        self.assertEqual(create_time, image['created_at'])
        self.assertEqual(create_time, image['updated_at'])
        self.assertTrue(uuid.UUID(image['id']))
        self.assertNotIn('tags', image)

    def test_image_create_duplicate_id(self):
        self.assertRaises(exception.Duplicate, self.db_api.image_create, self.context, {'id': UUID1, 'status': 'queued'})

    def test_image_create_with_locations(self):
        locations = [{'url': 'a', 'metadata': {}, 'status': 'active'}, {'url': 'b', 'metadata': {}, 'status': 'active'}]
        fixture = {'status': 'queued', 'locations': locations}
        image = self.db_api.image_create(self.context, fixture)
        actual = [{'url': location['url'], 'metadata': location['metadata'], 'status': location['status']} for location in image['locations']]
        self.assertEqual(locations, actual)

    def test_image_create_without_locations(self):
        locations = []
        fixture = {'status': 'queued', 'locations': locations}
        self.db_api.image_create(self.context, fixture)

    def test_image_create_with_location_data(self):
        location_data = [{'url': 'a', 'metadata': {'key': 'value'}, 'status': 'active'}, {'url': 'b', 'metadata': {}, 'status': 'active'}]
        fixture = {'status': 'queued', 'locations': location_data}
        image = self.db_api.image_create(self.context, fixture)
        actual = [{'url': location['url'], 'metadata': location['metadata'], 'status': location['status']} for location in image['locations']]
        self.assertEqual(location_data, actual)

    def test_image_create_properties(self):
        fixture = {'status': 'queued', 'properties': {'ping': 'pong'}}
        image = self.db_api.image_create(self.context, fixture)
        expected = [{'name': 'ping', 'value': 'pong'}]
        actual = [{'name': p['name'], 'value': p['value']} for p in image['properties']]
        self.assertEqual(expected, actual)

    def test_image_create_unknown_attributes(self):
        fixture = {'ping': 'pong'}
        self.assertRaises(exception.Invalid, self.db_api.image_create, self.context, fixture)

    def test_image_create_bad_name(self):
        bad_name = 'A name with forbidden symbol 😪'
        fixture = {'name': bad_name, 'size': 12, 'status': 'queued'}
        self.assertRaises(exception.Invalid, self.db_api.image_create, self.context, fixture)

    def test_image_create_bad_checksum(self):
        bad_checksum = '42' * 42
        fixture = {'checksum': bad_checksum}
        self.assertRaises(exception.Invalid, self.db_api.image_create, self.context, fixture)
        fixture = {'checksum': 'Я' * 32}
        self.assertRaises(exception.Invalid, self.db_api.image_create, self.context, fixture)

    def test_image_create_bad_int_params(self):
        int_too_long = 2 ** 31 + 42
        for param in ['min_disk', 'min_ram']:
            fixture = {param: int_too_long}
            self.assertRaises(exception.Invalid, self.db_api.image_create, self.context, fixture)

    def test_image_create_bad_property(self):
        fixture = {'status': 'queued', 'properties': {'bad': 'Bad 😪'}}
        self.assertRaises(exception.Invalid, self.db_api.image_create, self.context, fixture)
        fixture = {'status': 'queued', 'properties': {'Bad 😪': 'ok'}}
        self.assertRaises(exception.Invalid, self.db_api.image_create, self.context, fixture)

    def test_image_create_bad_location(self):
        location_data = [{'url': 'a', 'metadata': {'key': 'value'}, 'status': 'active'}, {'url': 'Bad 😊', 'metadata': {}, 'status': 'active'}]
        fixture = {'status': 'queued', 'locations': location_data}
        self.assertRaises(exception.Invalid, self.db_api.image_create, self.context, fixture)

    def test_image_update_core_attribute(self):
        fixture = {'status': 'queued'}
        image = self.db_api.image_update(self.adm_context, UUID3, fixture)
        self.assertEqual('queued', image['status'])
        self.assertNotEqual(image['created_at'], image['updated_at'])

    def test_image_update_with_locations(self):
        locations = [{'url': 'a', 'metadata': {}, 'status': 'active'}, {'url': 'b', 'metadata': {}, 'status': 'active'}]
        fixture = {'locations': locations}
        image = self.db_api.image_update(self.adm_context, UUID3, fixture)
        self.assertEqual(2, len(image['locations']))
        self.assertIn('id', image['locations'][0])
        self.assertIn('id', image['locations'][1])
        image['locations'][0].pop('id')
        image['locations'][1].pop('id')
        self.assertEqual(locations, image['locations'])

    def test_image_update_with_location_data(self):
        location_data = [{'url': 'a', 'metadata': {'key': 'value'}, 'status': 'active'}, {'url': 'b', 'metadata': {}, 'status': 'active'}]
        fixture = {'locations': location_data}
        image = self.db_api.image_update(self.adm_context, UUID3, fixture)
        self.assertEqual(2, len(image['locations']))
        self.assertIn('id', image['locations'][0])
        self.assertIn('id', image['locations'][1])
        image['locations'][0].pop('id')
        image['locations'][1].pop('id')
        self.assertEqual(location_data, image['locations'])

    def test_image_update(self):
        fixture = {'status': 'queued', 'properties': {'ping': 'pong'}}
        image = self.db_api.image_update(self.adm_context, UUID3, fixture)
        expected = [{'name': 'ping', 'value': 'pong'}]
        actual = [{'name': p['name'], 'value': p['value']} for p in image['properties']]
        self.assertEqual(expected, actual)
        self.assertEqual('queued', image['status'])
        self.assertNotEqual(image['created_at'], image['updated_at'])

    def test_image_update_properties(self):
        fixture = {'properties': {'ping': 'pong'}}
        self.delay_inaccurate_clock()
        image = self.db_api.image_update(self.adm_context, UUID1, fixture)
        expected = {'ping': 'pong', 'foo': 'bar', 'far': 'boo'}
        actual = {p['name']: p['value'] for p in image['properties']}
        self.assertEqual(expected, actual)
        self.assertNotEqual(image['created_at'], image['updated_at'])

    def test_image_update_purge_properties(self):
        fixture = {'properties': {'ping': 'pong'}}
        image = self.db_api.image_update(self.adm_context, UUID1, fixture, purge_props=True)
        properties = {p['name']: p for p in image['properties']}
        self.assertIn('ping', properties)
        self.assertEqual('pong', properties['ping']['value'])
        self.assertFalse(properties['ping']['deleted'])
        self.assertIn('foo', properties)
        self.assertEqual('bar', properties['foo']['value'])
        self.assertTrue(properties['foo']['deleted'])

    def test_image_update_bad_name(self):
        fixture = {'name': 'A new name with forbidden symbol 😪'}
        self.assertRaises(exception.Invalid, self.db_api.image_update, self.adm_context, UUID1, fixture)

    def test_image_update_bad_property(self):
        fixture = {'status': 'queued', 'properties': {'bad': 'Bad 😪'}}
        self.assertRaises(exception.Invalid, self.db_api.image_update, self.adm_context, UUID1, fixture)
        fixture = {'status': 'queued', 'properties': {'Bad 😪': 'ok'}}
        self.assertRaises(exception.Invalid, self.db_api.image_update, self.adm_context, UUID1, fixture)

    def test_image_update_bad_location(self):
        location_data = [{'url': 'a', 'metadata': {'key': 'value'}, 'status': 'active'}, {'url': 'Bad 😊', 'metadata': {}, 'status': 'active'}]
        fixture = {'status': 'queued', 'locations': location_data}
        self.assertRaises(exception.Invalid, self.db_api.image_update, self.adm_context, UUID1, fixture)

    def test_update_locations_direct(self):
        """
        For some reasons update_locations can be called directly
        (not via image_update), so better check that everything is ok if passed
        4 byte unicode characters
        """
        location_data = [{'url': 'a', 'metadata': {'key': 'value'}, 'status': 'active'}]
        fixture = {'locations': location_data}
        image = self.db_api.image_update(self.adm_context, UUID1, fixture)
        self.assertEqual(1, len(image['locations']))
        self.assertIn('id', image['locations'][0])
        loc_id = image['locations'][0].pop('id')
        bad_location = {'url': 'Bad 😊', 'metadata': {}, 'status': 'active', 'id': loc_id}
        self.assertRaises(exception.Invalid, self.db_api.image_location_update, self.adm_context, UUID1, bad_location)

    def test_image_property_delete(self):
        fixture = {'name': 'ping', 'value': 'pong', 'image_id': UUID1}
        prop = self.db_api.image_property_create(self.context, fixture)
        prop = self.db_api.image_property_delete(self.context, prop['name'], UUID1)
        self.assertIsNotNone(prop['deleted_at'])
        self.assertTrue(prop['deleted'])

    def test_image_get(self):
        image = self.db_api.image_get(self.context, UUID1)
        self.assertEqual(self.fixtures[0]['id'], image['id'])

    def test_image_get_disallow_deleted(self):
        self.db_api.image_destroy(self.adm_context, UUID1)
        self.assertRaises(exception.NotFound, self.db_api.image_get, self.context, UUID1)

    def test_image_get_allow_deleted(self):
        self.db_api.image_destroy(self.adm_context, UUID1)
        image = self.db_api.image_get(self.adm_context, UUID1)
        self.assertEqual(self.fixtures[0]['id'], image['id'])
        self.assertTrue(image['deleted'])

    def test_image_get_force_allow_deleted(self):
        self.db_api.image_destroy(self.adm_context, UUID1)
        image = self.db_api.image_get(self.context, UUID1, force_show_deleted=True)
        self.assertEqual(self.fixtures[0]['id'], image['id'])

    def test_image_get_not_owned(self):
        TENANT1 = str(uuid.uuid4())
        TENANT2 = str(uuid.uuid4())
        ctxt1 = context.RequestContext(is_admin=False, tenant=TENANT1, auth_token='user:%s:user' % TENANT1)
        ctxt2 = context.RequestContext(is_admin=False, tenant=TENANT2, auth_token='user:%s:user' % TENANT2)
        image = self.db_api.image_create(ctxt1, {'status': 'queued', 'owner': TENANT1})
        self.assertRaises(exception.Forbidden, self.db_api.image_get, ctxt2, image['id'])

    def test_image_get_not_found(self):
        UUID = str(uuid.uuid4())
        self.assertRaises(exception.NotFound, self.db_api.image_get, self.context, UUID)

    def test_image_get_all(self):
        images = self.db_api.image_get_all(self.context)
        self.assertEqual(3, len(images))

    def test_image_get_all_with_filter(self):
        images = self.db_api.image_get_all(self.context, filters={'id': self.fixtures[0]['id']})
        self.assertEqual(1, len(images))
        self.assertEqual(self.fixtures[0]['id'], images[0]['id'])

    def test_image_get_all_with_filter_user_defined_property(self):
        images = self.db_api.image_get_all(self.context, filters={'foo': 'bar'})
        self.assertEqual(1, len(images))
        self.assertEqual(self.fixtures[0]['id'], images[0]['id'])

    def test_image_get_all_with_filter_nonexistent_userdef_property(self):
        images = self.db_api.image_get_all(self.context, filters={'faz': 'boo'})
        self.assertEqual(0, len(images))

    def test_image_get_all_with_filter_userdef_prop_nonexistent_value(self):
        images = self.db_api.image_get_all(self.context, filters={'foo': 'baz'})
        self.assertEqual(0, len(images))

    def test_image_get_all_with_filter_multiple_user_defined_properties(self):
        images = self.db_api.image_get_all(self.context, filters={'foo': 'bar', 'far': 'boo'})
        self.assertEqual(1, len(images))
        self.assertEqual(images[0]['id'], self.fixtures[0]['id'])

    def test_image_get_all_with_filter_nonexistent_user_defined_property(self):
        images = self.db_api.image_get_all(self.context, filters={'foo': 'bar', 'faz': 'boo'})
        self.assertEqual(0, len(images))

    def test_image_get_all_with_filter_user_deleted_property(self):
        fixture = {'name': 'poo', 'value': 'bear', 'image_id': UUID1}
        prop = self.db_api.image_property_create(self.context, fixture)
        images = self.db_api.image_get_all(self.context, filters={'properties': {'poo': 'bear'}})
        self.assertEqual(1, len(images))
        self.db_api.image_property_delete(self.context, prop['name'], images[0]['id'])
        images = self.db_api.image_get_all(self.context, filters={'properties': {'poo': 'bear'}})
        self.assertEqual(0, len(images))

    def test_image_get_all_with_filter_undefined_property(self):
        images = self.db_api.image_get_all(self.context, filters={'poo': 'bear'})
        self.assertEqual(0, len(images))

    def test_image_get_all_with_filter_protected(self):
        images = self.db_api.image_get_all(self.context, filters={'protected': True})
        self.assertEqual(1, len(images))
        images = self.db_api.image_get_all(self.context, filters={'protected': False})
        self.assertEqual(2, len(images))

    def test_image_get_all_with_filter_comparative_created_at(self):
        anchor = timeutils.isotime(self.fixtures[0]['created_at'])
        time_expr = 'lt:' + anchor
        images = self.db_api.image_get_all(self.context, filters={'created_at': time_expr})
        self.assertEqual(0, len(images))

    def test_image_get_all_with_filter_comparative_updated_at(self):
        anchor = timeutils.isotime(self.fixtures[0]['updated_at'])
        time_expr = 'lt:' + anchor
        images = self.db_api.image_get_all(self.context, filters={'updated_at': time_expr})
        self.assertEqual(0, len(images))

    def test_filter_image_by_invalid_operator(self):
        self.assertRaises(exception.InvalidFilterOperatorValue, self.db_api.image_get_all, self.context, filters={'status': 'lala:active'})

    def test_image_get_all_with_filter_in_status(self):
        images = self.db_api.image_get_all(self.context, filters={'status': 'in:active'})
        self.assertEqual(3, len(images))

    def test_image_get_all_with_filter_in_name(self):
        data = 'in:%s' % self.fixtures[0]['name']
        images = self.db_api.image_get_all(self.context, filters={'name': data})
        self.assertEqual(3, len(images))

    def test_image_get_all_with_filter_in_container_format(self):
        images = self.db_api.image_get_all(self.context, filters={'container_format': 'in:ami,bare,ovf'})
        self.assertEqual(3, len(images))

    def test_image_get_all_with_filter_in_disk_format(self):
        images = self.db_api.image_get_all(self.context, filters={'disk_format': 'in:vhd'})
        self.assertEqual(3, len(images))

    def test_image_get_all_with_filter_in_id(self):
        data = 'in:%s,%s' % (UUID1, UUID2)
        images = self.db_api.image_get_all(self.context, filters={'id': data})
        self.assertEqual(2, len(images))

    def test_image_get_all_with_quotes(self):
        fixture = {'name': 'fake\\"name'}
        self.db_api.image_update(self.adm_context, UUID3, fixture)
        fixture = {'name': 'fake,name'}
        self.db_api.image_update(self.adm_context, UUID2, fixture)
        fixture = {'name': 'fakename'}
        self.db_api.image_update(self.adm_context, UUID1, fixture)
        data = 'in:"fake\\"name",fakename,"fake,name"'
        images = self.db_api.image_get_all(self.context, filters={'name': data})
        self.assertEqual(3, len(images))

    def test_image_get_all_with_invalid_quotes(self):
        invalid_expr = ['in:"name', 'in:"name"name', 'in:name"dd"', 'in:na"me', 'in:"name""name"']
        for expr in invalid_expr:
            self.assertRaises(exception.InvalidParameterValue, self.db_api.image_get_all, self.context, filters={'name': expr})

    def test_image_get_all_size_min_max(self):
        images = self.db_api.image_get_all(self.context, filters={'size_min': 10, 'size_max': 15})
        self.assertEqual(1, len(images))
        self.assertEqual(self.fixtures[0]['id'], images[0]['id'])

    def test_image_get_all_size_min(self):
        images = self.db_api.image_get_all(self.context, filters={'size_min': 15})
        self.assertEqual(2, len(images))
        self.assertEqual(self.fixtures[2]['id'], images[0]['id'])
        self.assertEqual(self.fixtures[1]['id'], images[1]['id'])

    def test_image_get_all_size_range(self):
        images = self.db_api.image_get_all(self.context, filters={'size_max': 15, 'size_min': 20})
        self.assertEqual(0, len(images))

    def test_image_get_all_size_max(self):
        images = self.db_api.image_get_all(self.context, filters={'size_max': 15})
        self.assertEqual(1, len(images))
        self.assertEqual(self.fixtures[0]['id'], images[0]['id'])

    def test_image_get_all_with_filter_min_range_bad_value(self):
        self.assertRaises(exception.InvalidFilterRangeValue, self.db_api.image_get_all, self.context, filters={'size_min': 'blah'})

    def test_image_get_all_with_filter_max_range_bad_value(self):
        self.assertRaises(exception.InvalidFilterRangeValue, self.db_api.image_get_all, self.context, filters={'size_max': 'blah'})

    def test_image_get_all_marker(self):
        images = self.db_api.image_get_all(self.context, marker=UUID3)
        self.assertEqual(2, len(images))

    def test_image_get_all_marker_with_size(self):
        images = self.db_api.image_get_all(self.context, sort_key=['size'], marker=UUID3)
        self.assertEqual(2, len(images))
        self.assertEqual(17, images[0]['size'])
        self.assertEqual(13, images[1]['size'])

    def test_image_get_all_marker_deleted(self):
        """Cannot specify a deleted image as a marker."""
        self.db_api.image_destroy(self.adm_context, UUID1)
        filters = {'deleted': False}
        self.assertRaises(exception.NotFound, self.db_api.image_get_all, self.context, marker=UUID1, filters=filters)

    def test_image_get_all_marker_deleted_showing_deleted_as_admin(self):
        """Specify a deleted image as a marker if showing deleted images."""
        self.db_api.image_destroy(self.adm_context, UUID3)
        images = self.db_api.image_get_all(self.adm_context, marker=UUID3)
        self.assertEqual(2, len(images))

    def test_image_get_all_marker_deleted_showing_deleted(self):
        """Specify a deleted image as a marker if showing deleted images.

        A non-admin user has to explicitly ask for deleted
        images, and should only see deleted images in the results
        """
        self.db_api.image_destroy(self.adm_context, UUID3)
        self.db_api.image_destroy(self.adm_context, UUID1)
        filters = {'deleted': True}
        images = self.db_api.image_get_all(self.context, marker=UUID3, filters=filters)
        self.assertEqual(1, len(images))

    def test_image_get_all_marker_null_name_desc(self):
        """Check an image with name null is handled

        Check an image with name null is handled
        marker is specified and order is descending
        """
        TENANT1 = str(uuid.uuid4())
        ctxt1 = context.RequestContext(is_admin=False, tenant=TENANT1, auth_token='user:%s:user' % TENANT1)
        UUIDX = str(uuid.uuid4())
        self.db_api.image_create(ctxt1, {'id': UUIDX, 'status': 'queued', 'name': None, 'owner': TENANT1})
        images = self.db_api.image_get_all(ctxt1, marker=UUIDX, sort_key=['name'], sort_dir=['desc'])
        image_ids = [image['id'] for image in images]
        expected = []
        self.assertEqual(sorted(expected), sorted(image_ids))

    def test_image_get_all_marker_null_disk_format_desc(self):
        """Check an image with disk_format null is handled

        Check an image with disk_format null is handled when
        marker is specified and order is descending
        """
        TENANT1 = str(uuid.uuid4())
        ctxt1 = context.RequestContext(is_admin=False, tenant=TENANT1, auth_token='user:%s:user' % TENANT1)
        UUIDX = str(uuid.uuid4())
        self.db_api.image_create(ctxt1, {'id': UUIDX, 'status': 'queued', 'disk_format': None, 'owner': TENANT1})
        images = self.db_api.image_get_all(ctxt1, marker=UUIDX, sort_key=['disk_format'], sort_dir=['desc'])
        image_ids = [image['id'] for image in images]
        expected = []
        self.assertEqual(sorted(expected), sorted(image_ids))

    def test_image_get_all_marker_null_container_format_desc(self):
        """Check an image with container_format null is handled

        Check an image with container_format null is handled when
        marker is specified and order is descending
        """
        TENANT1 = str(uuid.uuid4())
        ctxt1 = context.RequestContext(is_admin=False, tenant=TENANT1, auth_token='user:%s:user' % TENANT1)
        UUIDX = str(uuid.uuid4())
        self.db_api.image_create(ctxt1, {'id': UUIDX, 'status': 'queued', 'container_format': None, 'owner': TENANT1})
        images = self.db_api.image_get_all(ctxt1, marker=UUIDX, sort_key=['container_format'], sort_dir=['desc'])
        image_ids = [image['id'] for image in images]
        expected = []
        self.assertEqual(sorted(expected), sorted(image_ids))

    def test_image_get_all_marker_null_name_asc(self):
        """Check an image with name null is handled

        Check an image with name null is handled when
        marker is specified and order is ascending
        """
        TENANT1 = str(uuid.uuid4())
        ctxt1 = context.RequestContext(is_admin=False, tenant=TENANT1, auth_token='user:%s:user' % TENANT1)
        UUIDX = str(uuid.uuid4())
        self.db_api.image_create(ctxt1, {'id': UUIDX, 'status': 'queued', 'name': None, 'owner': TENANT1})
        images = self.db_api.image_get_all(ctxt1, marker=UUIDX, sort_key=['name'], sort_dir=['asc'])
        image_ids = [image['id'] for image in images]
        expected = [UUID3, UUID2, UUID1]
        self.assertEqual(sorted(expected), sorted(image_ids))

    def test_image_get_all_marker_null_disk_format_asc(self):
        """Check an image with disk_format null is handled

        Check an image with disk_format null is handled when
        marker is specified and order is ascending
        """
        TENANT1 = str(uuid.uuid4())
        ctxt1 = context.RequestContext(is_admin=False, tenant=TENANT1, auth_token='user:%s:user' % TENANT1)
        UUIDX = str(uuid.uuid4())
        self.db_api.image_create(ctxt1, {'id': UUIDX, 'status': 'queued', 'disk_format': None, 'owner': TENANT1})
        images = self.db_api.image_get_all(ctxt1, marker=UUIDX, sort_key=['disk_format'], sort_dir=['asc'])
        image_ids = [image['id'] for image in images]
        expected = [UUID3, UUID2, UUID1]
        self.assertEqual(sorted(expected), sorted(image_ids))

    def test_image_get_all_marker_null_container_format_asc(self):
        """Check an image with container_format null is handled

        Check an image with container_format null is handled when
        marker is specified and order is ascending
        """
        TENANT1 = str(uuid.uuid4())
        ctxt1 = context.RequestContext(is_admin=False, tenant=TENANT1, auth_token='user:%s:user' % TENANT1)
        UUIDX = str(uuid.uuid4())
        self.db_api.image_create(ctxt1, {'id': UUIDX, 'status': 'queued', 'container_format': None, 'owner': TENANT1})
        images = self.db_api.image_get_all(ctxt1, marker=UUIDX, sort_key=['container_format'], sort_dir=['asc'])
        image_ids = [image['id'] for image in images]
        expected = [UUID3, UUID2, UUID1]
        self.assertEqual(sorted(expected), sorted(image_ids))

    def test_image_get_all_limit(self):
        images = self.db_api.image_get_all(self.context, limit=2)
        self.assertEqual(2, len(images))
        images = self.db_api.image_get_all(self.context, limit=None)
        self.assertEqual(3, len(images))
        images = self.db_api.image_get_all(self.context, limit=0)
        self.assertEqual(0, len(images))

    def test_image_get_all_owned(self):
        TENANT1 = str(uuid.uuid4())
        ctxt1 = context.RequestContext(is_admin=False, tenant=TENANT1, auth_token='user:%s:user' % TENANT1)
        UUIDX = str(uuid.uuid4())
        image_meta_data = {'id': UUIDX, 'status': 'queued', 'owner': TENANT1}
        self.db_api.image_create(ctxt1, image_meta_data)
        TENANT2 = str(uuid.uuid4())
        ctxt2 = context.RequestContext(is_admin=False, tenant=TENANT2, auth_token='user:%s:user' % TENANT2)
        UUIDY = str(uuid.uuid4())
        image_meta_data = {'id': UUIDY, 'status': 'queued', 'owner': TENANT2}
        self.db_api.image_create(ctxt2, image_meta_data)
        images = self.db_api.image_get_all(ctxt1)
        image_ids = [image['id'] for image in images]
        expected = [UUIDX, UUID3, UUID2, UUID1]
        self.assertEqual(sorted(expected), sorted(image_ids))

    def test_image_get_all_owned_checksum(self):
        TENANT1 = str(uuid.uuid4())
        ctxt1 = context.RequestContext(is_admin=False, tenant=TENANT1, auth_token='user:%s:user' % TENANT1)
        UUIDX = str(uuid.uuid4())
        CHECKSUM1 = '91264c3edf5972c9f1cb309543d38a5c'
        image_meta_data = {'id': UUIDX, 'status': 'queued', 'checksum': CHECKSUM1, 'owner': TENANT1}
        self.db_api.image_create(ctxt1, image_meta_data)
        image_member_data = {'image_id': UUIDX, 'member': TENANT1, 'can_share': False, 'status': 'accepted'}
        self.db_api.image_member_create(ctxt1, image_member_data)
        TENANT2 = str(uuid.uuid4())
        ctxt2 = context.RequestContext(is_admin=False, tenant=TENANT2, auth_token='user:%s:user' % TENANT2)
        UUIDY = str(uuid.uuid4())
        CHECKSUM2 = '92264c3edf5972c9f1cb309543d38a5c'
        image_meta_data = {'id': UUIDY, 'status': 'queued', 'checksum': CHECKSUM2, 'owner': TENANT2}
        self.db_api.image_create(ctxt2, image_meta_data)
        image_member_data = {'image_id': UUIDY, 'member': TENANT2, 'can_share': False, 'status': 'accepted'}
        self.db_api.image_member_create(ctxt2, image_member_data)
        filters = {'visibility': 'shared', 'checksum': CHECKSUM2}
        images = self.db_api.image_get_all(ctxt2, filters)
        self.assertEqual(1, len(images))
        self.assertEqual(UUIDY, images[0]['id'])

    def test_image_get_all_with_filter_tags(self):
        self.db_api.image_tag_create(self.context, UUID1, 'x86')
        self.db_api.image_tag_create(self.context, UUID1, '64bit')
        self.db_api.image_tag_create(self.context, UUID2, 'power')
        self.db_api.image_tag_create(self.context, UUID2, '64bit')
        images = self.db_api.image_get_all(self.context, filters={'tags': ['64bit']})
        self.assertEqual(2, len(images))
        image_ids = [image['id'] for image in images]
        expected = [UUID1, UUID2]
        self.assertEqual(sorted(expected), sorted(image_ids))

    def test_image_get_all_with_filter_multi_tags(self):
        self.db_api.image_tag_create(self.context, UUID1, 'x86')
        self.db_api.image_tag_create(self.context, UUID1, '64bit')
        self.db_api.image_tag_create(self.context, UUID2, 'power')
        self.db_api.image_tag_create(self.context, UUID2, '64bit')
        images = self.db_api.image_get_all(self.context, filters={'tags': ['64bit', 'power']})
        self.assertEqual(1, len(images))
        self.assertEqual(UUID2, images[0]['id'])

    def test_image_get_all_with_filter_tags_and_nonexistent(self):
        self.db_api.image_tag_create(self.context, UUID1, 'x86')
        images = self.db_api.image_get_all(self.context, filters={'tags': ['x86', 'fake']})
        self.assertEqual(0, len(images))

    def test_image_get_all_with_filter_deleted_tags(self):
        tag = self.db_api.image_tag_create(self.context, UUID1, 'AIX')
        images = self.db_api.image_get_all(self.context, filters={'tags': [tag]})
        self.assertEqual(1, len(images))
        self.db_api.image_tag_delete(self.context, UUID1, tag)
        images = self.db_api.image_get_all(self.context, filters={'tags': [tag]})
        self.assertEqual(0, len(images))

    def test_image_get_all_with_filter_undefined_tags(self):
        images = self.db_api.image_get_all(self.context, filters={'tags': ['fake']})
        self.assertEqual(0, len(images))

    def test_image_paginate(self):
        """Paginate through a list of images using limit and marker"""
        now = timeutils.utcnow()
        extra_uuids = [(str(uuid.uuid4()), now + datetime.timedelta(seconds=i * 5)) for i in range(2)]
        extra_images = [build_image_fixture(id=_id, created_at=_dt, updated_at=_dt) for _id, _dt in extra_uuids]
        self.create_images(extra_images)
        extra_uuids.reverse()
        page = self.db_api.image_get_all(self.context, limit=2)
        self.assertEqual([i[0] for i in extra_uuids], [i['id'] for i in page])
        last = page[-1]['id']
        page = self.db_api.image_get_all(self.context, limit=2, marker=last)
        self.assertEqual([UUID3, UUID2], [i['id'] for i in page])
        page = self.db_api.image_get_all(self.context, limit=2, marker=UUID2)
        self.assertEqual([UUID1], [i['id'] for i in page])

    def test_image_get_all_invalid_sort_key(self):
        self.assertRaises(exception.InvalidSortKey, self.db_api.image_get_all, self.context, sort_key=['blah'])

    def test_image_get_all_limit_marker(self):
        images = self.db_api.image_get_all(self.context, limit=2)
        self.assertEqual(2, len(images))

    def test_image_get_all_with_tag_returning(self):
        expected_tags = {UUID1: ['foo'], UUID2: ['bar'], UUID3: ['baz']}
        self.db_api.image_tag_create(self.context, UUID1, expected_tags[UUID1][0])
        self.db_api.image_tag_create(self.context, UUID2, expected_tags[UUID2][0])
        self.db_api.image_tag_create(self.context, UUID3, expected_tags[UUID3][0])
        images = self.db_api.image_get_all(self.context, return_tag=True)
        self.assertEqual(3, len(images))
        for image in images:
            self.assertIn('tags', image)
            self.assertEqual(expected_tags[image['id']], image['tags'])
        self.db_api.image_tag_delete(self.context, UUID1, expected_tags[UUID1][0])
        expected_tags[UUID1] = []
        images = self.db_api.image_get_all(self.context, return_tag=True)
        self.assertEqual(3, len(images))
        for image in images:
            self.assertIn('tags', image)
            self.assertEqual(expected_tags[image['id']], image['tags'])

    def test_image_destroy(self):
        location_data = [{'url': 'a', 'metadata': {'key': 'value'}, 'status': 'active'}, {'url': 'b', 'metadata': {}, 'status': 'active'}]
        fixture = {'status': 'queued', 'locations': location_data}
        image = self.db_api.image_create(self.context, fixture)
        IMG_ID = image['id']
        fixture = {'name': 'ping', 'value': 'pong', 'image_id': IMG_ID}
        prop = self.db_api.image_property_create(self.context, fixture)
        TENANT2 = str(uuid.uuid4())
        fixture = {'image_id': IMG_ID, 'member': TENANT2, 'can_share': False}
        member = self.db_api.image_member_create(self.context, fixture)
        self.db_api.image_tag_create(self.context, IMG_ID, 'snarf')
        self.assertEqual(2, len(image['locations']))
        self.assertIn('id', image['locations'][0])
        self.assertIn('id', image['locations'][1])
        image['locations'][0].pop('id')
        image['locations'][1].pop('id')
        self.assertEqual(location_data, image['locations'])
        self.assertEqual(('ping', 'pong', IMG_ID, False), (prop['name'], prop['value'], prop['image_id'], prop['deleted']))
        self.assertEqual((TENANT2, IMG_ID, False), (member['member'], member['image_id'], member['can_share']))
        self.assertEqual(['snarf'], self.db_api.image_tag_get_all(self.context, IMG_ID))
        image = self.db_api.image_destroy(self.adm_context, IMG_ID)
        self.assertTrue(image['deleted'])
        self.assertTrue(image['deleted_at'])
        self.assertRaises(exception.NotFound, self.db_api.image_get, self.context, IMG_ID)
        self.assertEqual([], image['locations'])
        prop = image['properties'][0]
        self.assertEqual(('ping', IMG_ID, True), (prop['name'], prop['image_id'], prop['deleted']))
        self.context.auth_token = 'user:%s:user' % TENANT2
        members = self.db_api.image_member_find(self.context, IMG_ID)
        self.assertEqual([], members)
        tags = self.db_api.image_tag_get_all(self.context, IMG_ID)
        self.assertEqual([], tags)

    def test_image_destroy_with_delete_all(self):
        """Check the image child element's _image_delete_all methods.

        checks if all the image_delete_all methods deletes only the child
        elements of the image to be deleted.
        """
        TENANT2 = str(uuid.uuid4())
        location_data = [{'url': 'a', 'metadata': {'key': 'value'}, 'status': 'active'}, {'url': 'b', 'metadata': {}, 'status': 'active'}]

        def _create_image_with_child_entries():
            fixture = {'status': 'queued', 'locations': location_data}
            image_id = self.db_api.image_create(self.context, fixture)['id']
            fixture = {'name': 'ping', 'value': 'pong', 'image_id': image_id}
            self.db_api.image_property_create(self.context, fixture)
            fixture = {'image_id': image_id, 'member': TENANT2, 'can_share': False}
            self.db_api.image_member_create(self.context, fixture)
            self.db_api.image_tag_create(self.context, image_id, 'snarf')
            return image_id
        ACTIVE_IMG_ID = _create_image_with_child_entries()
        DEL_IMG_ID = _create_image_with_child_entries()
        deleted_image = self.db_api.image_destroy(self.adm_context, DEL_IMG_ID)
        self.assertTrue(deleted_image['deleted'])
        self.assertTrue(deleted_image['deleted_at'])
        self.assertRaises(exception.NotFound, self.db_api.image_get, self.context, DEL_IMG_ID)
        active_image = self.db_api.image_get(self.context, ACTIVE_IMG_ID)
        self.assertFalse(active_image['deleted'])
        self.assertFalse(active_image['deleted_at'])
        self.assertEqual(2, len(active_image['locations']))
        self.assertIn('id', active_image['locations'][0])
        self.assertIn('id', active_image['locations'][1])
        active_image['locations'][0].pop('id')
        active_image['locations'][1].pop('id')
        self.assertEqual(location_data, active_image['locations'])
        self.assertEqual(1, len(active_image['properties']))
        prop = active_image['properties'][0]
        self.assertEqual(('ping', 'pong', ACTIVE_IMG_ID), (prop['name'], prop['value'], prop['image_id']))
        self.assertEqual((False, None), (prop['deleted'], prop['deleted_at']))
        self.context.auth_token = 'user:%s:user' % TENANT2
        members = self.db_api.image_member_find(self.context, ACTIVE_IMG_ID)
        self.assertEqual(1, len(members))
        member = members[0]
        self.assertEqual((TENANT2, ACTIVE_IMG_ID, False), (member['member'], member['image_id'], member['can_share']))
        tags = self.db_api.image_tag_get_all(self.context, ACTIVE_IMG_ID)
        self.assertEqual(['snarf'], tags)

    def test_image_get_multiple_members(self):
        TENANT1 = str(uuid.uuid4())
        TENANT2 = str(uuid.uuid4())
        ctxt1 = context.RequestContext(is_admin=False, tenant=TENANT1, auth_token='user:%s:user' % TENANT1)
        ctxt2 = context.RequestContext(is_admin=False, tenant=TENANT2, auth_token='user:%s:user' % TENANT2)
        UUIDX = str(uuid.uuid4())
        self.db_api.image_create(ctxt1, {'id': UUIDX, 'status': 'queued', 'is_public': False, 'owner': TENANT1})
        values = {'image_id': UUIDX, 'member': TENANT2, 'can_share': False}
        self.db_api.image_member_create(ctxt1, values)
        image = self.db_api.image_get(ctxt2, UUIDX)
        self.assertEqual(UUIDX, image['id'])
        images = self.db_api.image_get_all(ctxt2)
        self.assertEqual(3, len(images))
        images = self.db_api.image_get_all(ctxt2, member_status='rejected')
        self.assertEqual(3, len(images))
        images = self.db_api.image_get_all(ctxt2, filters={'visibility': 'shared'})
        self.assertEqual(0, len(images))
        images = self.db_api.image_get_all(ctxt2, member_status='pending', filters={'visibility': 'shared'})
        self.assertEqual(1, len(images))
        images = self.db_api.image_get_all(ctxt2, member_status='all', filters={'visibility': 'shared'})
        self.assertEqual(1, len(images))
        images = self.db_api.image_get_all(ctxt2, member_status='pending')
        self.assertEqual(4, len(images))
        images = self.db_api.image_get_all(ctxt2, member_status='all')
        self.assertEqual(4, len(images))

    def test_is_image_visible(self):
        TENANT1 = str(uuid.uuid4())
        TENANT2 = str(uuid.uuid4())
        ctxt1 = context.RequestContext(is_admin=False, tenant=TENANT1, auth_token='user:%s:user' % TENANT1)
        ctxt2 = context.RequestContext(is_admin=False, tenant=TENANT2, auth_token='user:%s:user' % TENANT2)
        UUIDX = str(uuid.uuid4())
        image = self.db_api.image_create(ctxt1, {'id': UUIDX, 'status': 'queued', 'is_public': False, 'owner': TENANT1})
        values = {'image_id': UUIDX, 'member': TENANT2, 'can_share': False}
        self.db_api.image_member_create(ctxt1, values)
        result = self.db_api.is_image_visible(ctxt2, image)
        self.assertTrue(result)
        members = self.db_api.image_member_find(ctxt1, image_id=UUIDX)
        self.db_api.image_member_delete(ctxt1, members[0]['id'])
        result = self.db_api.is_image_visible(ctxt2, image)
        self.assertFalse(result)

    def test_is_community_image_visible(self):
        TENANT1 = str(uuid.uuid4())
        TENANT2 = str(uuid.uuid4())
        owners_ctxt = context.RequestContext(is_admin=False, tenant=TENANT1, auth_token='user:%s:user' % TENANT1)
        viewing_ctxt = context.RequestContext(is_admin=False, user=TENANT2, auth_token='user:%s:user' % TENANT2)
        UUIDX = str(uuid.uuid4())
        image = self.db_api.image_create(owners_ctxt, {'id': UUIDX, 'status': 'queued', 'visibility': 'community', 'owner': TENANT1})
        result = self.db_api.is_image_visible(owners_ctxt, image)
        self.assertTrue(result)
        result = self.db_api.is_image_visible(viewing_ctxt, image)
        self.assertTrue(result)

    def test_image_tag_create(self):
        tag = self.db_api.image_tag_create(self.context, UUID1, 'snap')
        self.assertEqual('snap', tag)

    def test_image_tag_create_bad_value(self):
        self.assertRaises(exception.Invalid, self.db_api.image_tag_create, self.context, UUID1, 'Bad 😪')

    def test_image_tag_set_all(self):
        tags = self.db_api.image_tag_get_all(self.context, UUID1)
        self.assertEqual([], tags)
        self.db_api.image_tag_set_all(self.context, UUID1, ['ping', 'pong'])
        tags = self.db_api.image_tag_get_all(self.context, UUID1)
        self.assertEqual(['ping', 'pong'], tags)

    def test_image_tag_get_all(self):
        self.db_api.image_tag_create(self.context, UUID1, 'snap')
        self.db_api.image_tag_create(self.context, UUID1, 'snarf')
        self.db_api.image_tag_create(self.context, UUID2, 'snarf')
        tags = self.db_api.image_tag_get_all(self.context, UUID1)
        expected = ['snap', 'snarf']
        self.assertEqual(expected, tags)
        tags = self.db_api.image_tag_get_all(self.context, UUID2)
        expected = ['snarf']
        self.assertEqual(expected, tags)

    def test_image_tag_get_all_no_tags(self):
        actual = self.db_api.image_tag_get_all(self.context, UUID1)
        self.assertEqual([], actual)

    def test_image_tag_get_all_non_existent_image(self):
        bad_image_id = str(uuid.uuid4())
        actual = self.db_api.image_tag_get_all(self.context, bad_image_id)
        self.assertEqual([], actual)

    def test_image_tag_delete(self):
        self.db_api.image_tag_create(self.context, UUID1, 'snap')
        self.db_api.image_tag_delete(self.context, UUID1, 'snap')
        self.assertRaises(exception.NotFound, self.db_api.image_tag_delete, self.context, UUID1, 'snap')

    @mock.patch.object(timeutils, 'utcnow')
    def test_image_member_create(self, mock_utcnow):
        mock_utcnow.return_value = datetime.datetime.utcnow()
        memberships = self.db_api.image_member_find(self.context)
        self.assertEqual([], memberships)
        TENANT1 = str(uuid.uuid4())
        self.context.auth_token = 'user:%s:user' % TENANT1
        self.db_api.image_member_create(self.context, {'member': TENANT1, 'image_id': UUID1})
        memberships = self.db_api.image_member_find(self.context)
        self.assertEqual(1, len(memberships))
        actual = memberships[0]
        self.assertIsNotNone(actual['created_at'])
        self.assertIsNotNone(actual['updated_at'])
        actual.pop('id')
        actual.pop('created_at')
        actual.pop('updated_at')
        expected = {'member': TENANT1, 'image_id': UUID1, 'can_share': False, 'status': 'pending', 'deleted': False}
        self.assertEqual(expected, actual)

    def test_image_member_update(self):
        TENANT1 = str(uuid.uuid4())
        self.context.auth_token = 'user:%s:user' % TENANT1
        member = self.db_api.image_member_create(self.context, {'member': TENANT1, 'image_id': UUID1})
        member_id = member.pop('id')
        member.pop('created_at')
        member.pop('updated_at')
        expected = {'member': TENANT1, 'image_id': UUID1, 'status': 'pending', 'can_share': False, 'deleted': False}
        self.assertEqual(expected, member)
        self.delay_inaccurate_clock()
        member = self.db_api.image_member_update(self.context, member_id, {'can_share': True})
        self.assertNotEqual(member['created_at'], member['updated_at'])
        member.pop('id')
        member.pop('created_at')
        member.pop('updated_at')
        expected = {'member': TENANT1, 'image_id': UUID1, 'status': 'pending', 'can_share': True, 'deleted': False}
        self.assertEqual(expected, member)
        members = self.db_api.image_member_find(self.context, member=TENANT1, image_id=UUID1)
        member = members[0]
        member.pop('id')
        member.pop('created_at')
        member.pop('updated_at')
        self.assertEqual(expected, member)

    def test_image_member_update_status(self):
        TENANT1 = str(uuid.uuid4())
        self.context.auth_token = 'user:%s:user' % TENANT1
        member = self.db_api.image_member_create(self.context, {'member': TENANT1, 'image_id': UUID1})
        member_id = member.pop('id')
        member.pop('created_at')
        member.pop('updated_at')
        expected = {'member': TENANT1, 'image_id': UUID1, 'status': 'pending', 'can_share': False, 'deleted': False}
        self.assertEqual(expected, member)
        self.delay_inaccurate_clock()
        member = self.db_api.image_member_update(self.context, member_id, {'status': 'accepted'})
        self.assertNotEqual(member['created_at'], member['updated_at'])
        member.pop('id')
        member.pop('created_at')
        member.pop('updated_at')
        expected = {'member': TENANT1, 'image_id': UUID1, 'status': 'accepted', 'can_share': False, 'deleted': False}
        self.assertEqual(expected, member)
        members = self.db_api.image_member_find(self.context, member=TENANT1, image_id=UUID1)
        member = members[0]
        member.pop('id')
        member.pop('created_at')
        member.pop('updated_at')
        self.assertEqual(expected, member)

    def test_image_member_find(self):
        TENANT1 = str(uuid.uuid4())
        TENANT2 = str(uuid.uuid4())
        fixtures = [{'member': TENANT1, 'image_id': UUID1}, {'member': TENANT1, 'image_id': UUID2, 'status': 'rejected'}, {'member': TENANT2, 'image_id': UUID1, 'status': 'accepted'}]
        for f in fixtures:
            self.db_api.image_member_create(self.context, copy.deepcopy(f))

        def _simplify(output):
            return

        def _assertMemberListMatch(list1, list2):

            def _simple(x):
                return set([(o['member'], o['image_id']) for o in x])
            self.assertEqual(_simple(list1), _simple(list2))
        self.context.auth_token = 'user:%s:user' % TENANT1
        output = self.db_api.image_member_find(self.context, member=TENANT1)
        _assertMemberListMatch([fixtures[0], fixtures[1]], output)
        output = self.db_api.image_member_find(self.adm_context, image_id=UUID1)
        _assertMemberListMatch([fixtures[0], fixtures[2]], output)
        self.context.auth_token = 'user:%s:user' % TENANT2
        output = self.db_api.image_member_find(self.context, member=TENANT2, image_id=UUID1)
        _assertMemberListMatch([fixtures[2]], output)
        output = self.db_api.image_member_find(self.context, status='accepted')
        _assertMemberListMatch([fixtures[2]], output)
        self.context.auth_token = 'user:%s:user' % TENANT1
        output = self.db_api.image_member_find(self.context, status='rejected')
        _assertMemberListMatch([fixtures[1]], output)
        output = self.db_api.image_member_find(self.context, status='pending')
        _assertMemberListMatch([fixtures[0]], output)
        output = self.db_api.image_member_find(self.context, status='pending', image_id=UUID2)
        _assertMemberListMatch([], output)
        image_id = str(uuid.uuid4())
        output = self.db_api.image_member_find(self.context, member=TENANT2, image_id=image_id)
        _assertMemberListMatch([], output)

    def test_image_member_count(self):
        TENANT1 = str(uuid.uuid4())
        self.db_api.image_member_create(self.context, {'member': TENANT1, 'image_id': UUID1})
        actual = self.db_api.image_member_count(self.context, UUID1)
        self.assertEqual(1, actual)

    def test_image_member_count_invalid_image_id(self):
        TENANT1 = str(uuid.uuid4())
        self.db_api.image_member_create(self.context, {'member': TENANT1, 'image_id': UUID1})
        self.assertRaises(exception.Invalid, self.db_api.image_member_count, self.context, None)

    def test_image_member_count_empty_image_id(self):
        TENANT1 = str(uuid.uuid4())
        self.db_api.image_member_create(self.context, {'member': TENANT1, 'image_id': UUID1})
        self.assertRaises(exception.Invalid, self.db_api.image_member_count, self.context, '')

    def test_image_member_delete(self):
        TENANT1 = str(uuid.uuid4())
        self.context.auth_token = 'user:%s:user' % TENANT1
        fixture = {'member': TENANT1, 'image_id': UUID1, 'can_share': True}
        member = self.db_api.image_member_create(self.context, fixture)
        self.assertEqual(1, len(self.db_api.image_member_find(self.context)))
        member = self.db_api.image_member_delete(self.context, member['id'])
        self.assertEqual(0, len(self.db_api.image_member_find(self.context)))