import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
class CatalogTests(object):
    _legacy_endpoint_id_in_endpoint = True
    _enabled_default_to_true_when_creating_endpoint = False

    def test_region_crud(self):
        region_id = 'default'
        new_region = unit.new_region_ref(id=region_id)
        res = PROVIDERS.catalog_api.create_region(new_region)
        expected_region = new_region.copy()
        expected_region['parent_region_id'] = None
        self.assertDictEqual(expected_region, res)
        parent_region_id = region_id
        new_region = unit.new_region_ref(parent_region_id=parent_region_id)
        region_id = new_region['id']
        res = PROVIDERS.catalog_api.create_region(new_region)
        self.assertDictEqual(new_region, res)
        regions = PROVIDERS.catalog_api.list_regions()
        self.assertThat(regions, matchers.HasLength(2))
        region_ids = [x['id'] for x in regions]
        self.assertIn(parent_region_id, region_ids)
        self.assertIn(region_id, region_ids)
        region_desc_update = {'description': uuid.uuid4().hex}
        res = PROVIDERS.catalog_api.update_region(region_id, region_desc_update)
        expected_region = new_region.copy()
        expected_region['description'] = region_desc_update['description']
        self.assertDictEqual(expected_region, res)
        PROVIDERS.catalog_api.delete_region(parent_region_id)
        self.assertRaises(exception.RegionNotFound, PROVIDERS.catalog_api.delete_region, parent_region_id)
        self.assertRaises(exception.RegionNotFound, PROVIDERS.catalog_api.get_region, parent_region_id)
        self.assertRaises(exception.RegionNotFound, PROVIDERS.catalog_api.get_region, region_id)

    def _create_region_with_parent_id(self, parent_id=None):
        new_region = unit.new_region_ref(parent_region_id=parent_id)
        PROVIDERS.catalog_api.create_region(new_region)
        return new_region

    def test_list_regions_filtered_by_parent_region_id(self):
        new_region = self._create_region_with_parent_id()
        parent_id = new_region['id']
        new_region = self._create_region_with_parent_id(parent_id)
        new_region = self._create_region_with_parent_id(parent_id)
        hints = driver_hints.Hints()
        hints.add_filter('parent_region_id', parent_id)
        regions = PROVIDERS.catalog_api.list_regions(hints)
        for region in regions:
            self.assertEqual(parent_id, region['parent_region_id'])

    @unit.skip_if_cache_disabled('catalog')
    def test_cache_layer_region_crud(self):
        new_region = unit.new_region_ref()
        region_id = new_region['id']
        PROVIDERS.catalog_api.create_region(new_region.copy())
        updated_region = copy.deepcopy(new_region)
        updated_region['description'] = uuid.uuid4().hex
        PROVIDERS.catalog_api.get_region(region_id)
        PROVIDERS.catalog_api.driver.update_region(region_id, updated_region)
        self.assertLessEqual(new_region.items(), PROVIDERS.catalog_api.get_region(region_id).items())
        PROVIDERS.catalog_api.get_region.invalidate(PROVIDERS.catalog_api, region_id)
        self.assertLessEqual(updated_region.items(), PROVIDERS.catalog_api.get_region(region_id).items())
        PROVIDERS.catalog_api.driver.delete_region(region_id)
        self.assertLessEqual(updated_region.items(), PROVIDERS.catalog_api.get_region(region_id).items())
        PROVIDERS.catalog_api.get_region.invalidate(PROVIDERS.catalog_api, region_id)
        self.assertRaises(exception.RegionNotFound, PROVIDERS.catalog_api.get_region, region_id)

    @unit.skip_if_cache_disabled('catalog')
    def test_invalidate_cache_when_updating_region(self):
        new_region = unit.new_region_ref()
        region_id = new_region['id']
        PROVIDERS.catalog_api.create_region(new_region)
        PROVIDERS.catalog_api.get_region(region_id)
        new_description = {'description': uuid.uuid4().hex}
        PROVIDERS.catalog_api.update_region(region_id, new_description)
        current_region = PROVIDERS.catalog_api.get_region(region_id)
        self.assertEqual(new_description['description'], current_region['description'])

    def test_update_region_extras(self):
        new_region = unit.new_region_ref()
        region_id = new_region['id']
        PROVIDERS.catalog_api.create_region(new_region)
        email = 'keystone@openstack.org'
        new_ref = {'description': uuid.uuid4().hex, 'email': email}
        PROVIDERS.catalog_api.update_region(region_id, new_ref)
        current_region = PROVIDERS.catalog_api.get_region(region_id)
        self.assertEqual(email, current_region['email'])

    def test_create_region_with_duplicate_id(self):
        new_region = unit.new_region_ref()
        PROVIDERS.catalog_api.create_region(new_region)
        self.assertRaises(exception.Conflict, PROVIDERS.catalog_api.create_region, new_region)

    def test_get_region_returns_not_found(self):
        self.assertRaises(exception.RegionNotFound, PROVIDERS.catalog_api.get_region, uuid.uuid4().hex)

    def test_delete_region_returns_not_found(self):
        self.assertRaises(exception.RegionNotFound, PROVIDERS.catalog_api.delete_region, uuid.uuid4().hex)

    def test_create_region_invalid_parent_region_returns_not_found(self):
        new_region = unit.new_region_ref(parent_region_id=uuid.uuid4().hex)
        self.assertRaises(exception.RegionNotFound, PROVIDERS.catalog_api.create_region, new_region)

    def test_avoid_creating_circular_references_in_regions_update(self):
        region_one = self._create_region_with_parent_id()
        self.assertRaises(exception.CircularRegionHierarchyError, PROVIDERS.catalog_api.update_region, region_one['id'], {'parent_region_id': region_one['id']})
        region_two = self._create_region_with_parent_id(region_one['id'])
        self.assertRaises(exception.CircularRegionHierarchyError, PROVIDERS.catalog_api.update_region, region_one['id'], {'parent_region_id': region_two['id']})
        region_three = self._create_region_with_parent_id(region_two['id'])
        region_four = self._create_region_with_parent_id(region_three['id'])
        self.assertRaises(exception.CircularRegionHierarchyError, PROVIDERS.catalog_api.update_region, region_two['id'], {'parent_region_id': region_four['id']})

    @mock.patch.object(base.CatalogDriverBase, '_ensure_no_circle_in_hierarchical_regions')
    def test_circular_regions_can_be_deleted(self, mock_ensure_on_circle):
        mock_ensure_on_circle.return_value = None
        region_one = self._create_region_with_parent_id()
        PROVIDERS.catalog_api.update_region(region_one['id'], {'parent_region_id': region_one['id']})
        PROVIDERS.catalog_api.delete_region(region_one['id'])
        self.assertRaises(exception.RegionNotFound, PROVIDERS.catalog_api.get_region, region_one['id'])
        region_one = self._create_region_with_parent_id()
        region_two = self._create_region_with_parent_id(region_one['id'])
        PROVIDERS.catalog_api.update_region(region_one['id'], {'parent_region_id': region_two['id']})
        PROVIDERS.catalog_api.delete_region(region_one['id'])
        self.assertRaises(exception.RegionNotFound, PROVIDERS.catalog_api.get_region, region_one['id'])
        self.assertRaises(exception.RegionNotFound, PROVIDERS.catalog_api.get_region, region_two['id'])
        region_one = self._create_region_with_parent_id()
        region_two = self._create_region_with_parent_id(region_one['id'])
        region_three = self._create_region_with_parent_id(region_two['id'])
        PROVIDERS.catalog_api.update_region(region_one['id'], {'parent_region_id': region_three['id']})
        PROVIDERS.catalog_api.delete_region(region_two['id'])
        self.assertRaises(exception.RegionNotFound, PROVIDERS.catalog_api.get_region, region_two['id'])
        self.assertRaises(exception.RegionNotFound, PROVIDERS.catalog_api.get_region, region_one['id'])
        self.assertRaises(exception.RegionNotFound, PROVIDERS.catalog_api.get_region, region_three['id'])

    def test_service_crud(self):
        new_service = unit.new_service_ref()
        service_id = new_service['id']
        res = PROVIDERS.catalog_api.create_service(service_id, new_service)
        self.assertDictEqual(new_service, res)
        services = PROVIDERS.catalog_api.list_services()
        self.assertIn(service_id, [x['id'] for x in services])
        service_name_update = {'name': uuid.uuid4().hex}
        res = PROVIDERS.catalog_api.update_service(service_id, service_name_update)
        expected_service = new_service.copy()
        expected_service['name'] = service_name_update['name']
        self.assertDictEqual(expected_service, res)
        PROVIDERS.catalog_api.delete_service(service_id)
        self.assertRaises(exception.ServiceNotFound, PROVIDERS.catalog_api.delete_service, service_id)
        self.assertRaises(exception.ServiceNotFound, PROVIDERS.catalog_api.get_service, service_id)

    def _create_random_service(self):
        new_service = unit.new_service_ref()
        service_id = new_service['id']
        return PROVIDERS.catalog_api.create_service(service_id, new_service)

    def test_service_filtering(self):
        target_service = self._create_random_service()
        unrelated_service1 = self._create_random_service()
        unrelated_service2 = self._create_random_service()
        hint_for_type = driver_hints.Hints()
        hint_for_type.add_filter(name='type', value=target_service['type'])
        services = PROVIDERS.catalog_api.list_services(hint_for_type)
        self.assertEqual(1, len(services))
        filtered_service = services[0]
        self.assertEqual(target_service['type'], filtered_service['type'])
        self.assertEqual(target_service['id'], filtered_service['id'])
        self.assertEqual(0, len(hint_for_type.filters))
        hint_for_name = driver_hints.Hints()
        hint_for_name.add_filter(name='name', value=target_service['name'])
        services = PROVIDERS.catalog_api.list_services(hint_for_name)
        self.assertEqual(3, len(services))
        self.assertEqual(1, len(hint_for_name.filters))
        PROVIDERS.catalog_api.delete_service(target_service['id'])
        PROVIDERS.catalog_api.delete_service(unrelated_service1['id'])
        PROVIDERS.catalog_api.delete_service(unrelated_service2['id'])

    @unit.skip_if_cache_disabled('catalog')
    def test_cache_layer_service_crud(self):
        new_service = unit.new_service_ref()
        service_id = new_service['id']
        res = PROVIDERS.catalog_api.create_service(service_id, new_service)
        self.assertDictEqual(new_service, res)
        PROVIDERS.catalog_api.get_service(service_id)
        updated_service = copy.deepcopy(new_service)
        updated_service['description'] = uuid.uuid4().hex
        PROVIDERS.catalog_api.driver.update_service(service_id, updated_service)
        self.assertLessEqual(new_service.items(), PROVIDERS.catalog_api.get_service(service_id).items())
        PROVIDERS.catalog_api.get_service.invalidate(PROVIDERS.catalog_api, service_id)
        self.assertLessEqual(updated_service.items(), PROVIDERS.catalog_api.get_service(service_id).items())
        PROVIDERS.catalog_api.driver.delete_service(service_id)
        self.assertLessEqual(updated_service.items(), PROVIDERS.catalog_api.get_service(service_id).items())
        PROVIDERS.catalog_api.get_service.invalidate(PROVIDERS.catalog_api, service_id)
        self.assertRaises(exception.ServiceNotFound, PROVIDERS.catalog_api.delete_service, service_id)
        self.assertRaises(exception.ServiceNotFound, PROVIDERS.catalog_api.get_service, service_id)

    @unit.skip_if_cache_disabled('catalog')
    def test_invalidate_cache_when_updating_service(self):
        new_service = unit.new_service_ref()
        service_id = new_service['id']
        PROVIDERS.catalog_api.create_service(service_id, new_service)
        PROVIDERS.catalog_api.get_service(service_id)
        new_type = {'type': uuid.uuid4().hex}
        PROVIDERS.catalog_api.update_service(service_id, new_type)
        current_service = PROVIDERS.catalog_api.get_service(service_id)
        self.assertEqual(new_type['type'], current_service['type'])

    def test_delete_service_with_endpoint(self):
        service = unit.new_service_ref()
        PROVIDERS.catalog_api.create_service(service['id'], service)
        endpoint = unit.new_endpoint_ref(service_id=service['id'], region_id=None)
        PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
        PROVIDERS.catalog_api.delete_service(service['id'])
        self.assertRaises(exception.EndpointNotFound, PROVIDERS.catalog_api.get_endpoint, endpoint['id'])
        self.assertRaises(exception.EndpointNotFound, PROVIDERS.catalog_api.delete_endpoint, endpoint['id'])

    def test_cache_layer_delete_service_with_endpoint(self):
        service = unit.new_service_ref()
        PROVIDERS.catalog_api.create_service(service['id'], service)
        endpoint = unit.new_endpoint_ref(service_id=service['id'], region_id=None)
        PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
        PROVIDERS.catalog_api.get_service(service['id'])
        PROVIDERS.catalog_api.get_endpoint(endpoint['id'])
        PROVIDERS.catalog_api.driver.delete_service(service['id'])
        self.assertLessEqual(endpoint.items(), PROVIDERS.catalog_api.get_endpoint(endpoint['id']).items())
        self.assertLessEqual(service.items(), PROVIDERS.catalog_api.get_service(service['id']).items())
        PROVIDERS.catalog_api.get_endpoint.invalidate(PROVIDERS.catalog_api, endpoint['id'])
        self.assertRaises(exception.EndpointNotFound, PROVIDERS.catalog_api.get_endpoint, endpoint['id'])
        self.assertRaises(exception.EndpointNotFound, PROVIDERS.catalog_api.delete_endpoint, endpoint['id'])
        second_endpoint = unit.new_endpoint_ref(service_id=service['id'], region_id=None)
        PROVIDERS.catalog_api.create_service(service['id'], service)
        PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
        PROVIDERS.catalog_api.create_endpoint(second_endpoint['id'], second_endpoint)
        PROVIDERS.catalog_api.delete_service(service['id'])
        self.assertRaises(exception.EndpointNotFound, PROVIDERS.catalog_api.get_endpoint, endpoint['id'])
        self.assertRaises(exception.EndpointNotFound, PROVIDERS.catalog_api.delete_endpoint, endpoint['id'])
        self.assertRaises(exception.EndpointNotFound, PROVIDERS.catalog_api.get_endpoint, second_endpoint['id'])
        self.assertRaises(exception.EndpointNotFound, PROVIDERS.catalog_api.delete_endpoint, second_endpoint['id'])

    def test_get_service_returns_not_found(self):
        self.assertRaises(exception.ServiceNotFound, PROVIDERS.catalog_api.get_service, uuid.uuid4().hex)

    def test_delete_service_returns_not_found(self):
        self.assertRaises(exception.ServiceNotFound, PROVIDERS.catalog_api.delete_service, uuid.uuid4().hex)

    def test_create_endpoint_nonexistent_service(self):
        endpoint = unit.new_endpoint_ref(service_id=uuid.uuid4().hex, region_id=None)
        self.assertRaises(exception.ValidationError, PROVIDERS.catalog_api.create_endpoint, endpoint['id'], endpoint)

    def test_update_endpoint_nonexistent_service(self):
        dummy_service, enabled_endpoint, dummy_disabled_endpoint = self._create_endpoints()
        new_endpoint = unit.new_endpoint_ref(service_id=uuid.uuid4().hex)
        self.assertRaises(exception.ValidationError, PROVIDERS.catalog_api.update_endpoint, enabled_endpoint['id'], new_endpoint)

    def test_create_endpoint_nonexistent_region(self):
        service = unit.new_service_ref()
        PROVIDERS.catalog_api.create_service(service['id'], service)
        endpoint = unit.new_endpoint_ref(service_id=service['id'])
        self.assertRaises(exception.ValidationError, PROVIDERS.catalog_api.create_endpoint, endpoint['id'], endpoint)

    def test_update_endpoint_nonexistent_region(self):
        dummy_service, enabled_endpoint, dummy_disabled_endpoint = self._create_endpoints()
        new_endpoint = unit.new_endpoint_ref(service_id=uuid.uuid4().hex)
        self.assertRaises(exception.ValidationError, PROVIDERS.catalog_api.update_endpoint, enabled_endpoint['id'], new_endpoint)

    def test_get_endpoint_returns_not_found(self):
        self.assertRaises(exception.EndpointNotFound, PROVIDERS.catalog_api.get_endpoint, uuid.uuid4().hex)

    def test_delete_endpoint_returns_not_found(self):
        self.assertRaises(exception.EndpointNotFound, PROVIDERS.catalog_api.delete_endpoint, uuid.uuid4().hex)

    def test_create_endpoint(self):
        service = unit.new_service_ref()
        PROVIDERS.catalog_api.create_service(service['id'], service)
        endpoint = unit.new_endpoint_ref(service_id=service['id'], region_id=None)
        PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint.copy())

    def test_update_endpoint(self):
        dummy_service_ref, endpoint_ref, dummy_disabled_endpoint_ref = self._create_endpoints()
        res = PROVIDERS.catalog_api.update_endpoint(endpoint_ref['id'], {'interface': 'private'})
        expected_endpoint = endpoint_ref.copy()
        expected_endpoint['enabled'] = True
        expected_endpoint['interface'] = 'private'
        if self._legacy_endpoint_id_in_endpoint:
            expected_endpoint['legacy_endpoint_id'] = None
        if self._enabled_default_to_true_when_creating_endpoint:
            expected_endpoint['enabled'] = True
        self.assertDictEqual(expected_endpoint, res)

    def _create_endpoints(self):

        def create_endpoint(service_id, region, **kwargs):
            ref = unit.new_endpoint_ref(service_id=service_id, region_id=region, url='http://localhost/%s' % uuid.uuid4().hex, **kwargs)
            PROVIDERS.catalog_api.create_endpoint(ref['id'], ref)
            return ref
        service_ref = unit.new_service_ref()
        service_id = service_ref['id']
        PROVIDERS.catalog_api.create_service(service_id, service_ref)
        region = unit.new_region_ref()
        PROVIDERS.catalog_api.create_region(region)
        enabled_endpoint_ref = create_endpoint(service_id, region['id'])
        disabled_endpoint_ref = create_endpoint(service_id, region['id'], enabled=False, interface='internal')
        return (service_ref, enabled_endpoint_ref, disabled_endpoint_ref)

    def test_list_endpoints(self):
        service = unit.new_service_ref()
        PROVIDERS.catalog_api.create_service(service['id'], service)
        expected_ids = set([uuid.uuid4().hex for _ in range(3)])
        for endpoint_id in expected_ids:
            endpoint = unit.new_endpoint_ref(service_id=service['id'], id=endpoint_id, region_id=None)
            PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
        endpoints = PROVIDERS.catalog_api.list_endpoints()
        self.assertEqual(expected_ids, set((e['id'] for e in endpoints)))

    def test_get_v3_catalog_endpoint_disabled(self):
        """Get back only enabled endpoints when get the v3 catalog."""
        enabled_endpoint_ref = self._create_endpoints()[1]
        user_id = uuid.uuid4().hex
        catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, self.project_bar['id'])
        endpoint_ids = [x['id'] for x in catalog[0]['endpoints']]
        self.assertEqual([enabled_endpoint_ref['id']], endpoint_ids)

    @unit.skip_if_cache_disabled('catalog')
    def test_invalidate_cache_when_updating_endpoint(self):
        service = unit.new_service_ref()
        PROVIDERS.catalog_api.create_service(service['id'], service)
        endpoint = unit.new_endpoint_ref(service_id=service['id'], region_id=None)
        PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
        PROVIDERS.catalog_api.get_endpoint(endpoint['id'])
        new_url = {'url': uuid.uuid4().hex}
        PROVIDERS.catalog_api.update_endpoint(endpoint['id'], new_url)
        current_endpoint = PROVIDERS.catalog_api.get_endpoint(endpoint['id'])
        self.assertEqual(new_url['url'], current_endpoint['url'])