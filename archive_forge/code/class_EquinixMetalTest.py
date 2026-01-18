import sys
import json
import unittest
import libcloud.compute.drivers.equinixmetal
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, KeyPair
from libcloud.test.compute import TestCaseMixin
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.equinixmetal import EquinixMetalNodeDriver
class EquinixMetalTest(unittest.TestCase, TestCaseMixin):

    def setUp(self):
        EquinixMetalNodeDriver.connectionCls.conn_class = EquinixMetalMockHttp
        self.driver = EquinixMetalNodeDriver('foo')

    def test_list_nodes(self):
        nodes = self.driver.list_nodes('project-id')
        self.assertEqual(len(nodes), 1)
        node = nodes[0]
        self.assertEqual(node.id, '1e52437e-bbbb-cccc-dddd-74a9dfd3d3bb')
        self.assertEqual(node.name, 'test-node')
        self.assertEqual(node.state, NodeState.RUNNING)
        self.assertTrue('147.75.255.255' in node.public_ips)
        self.assertTrue('2604:EEEE::EE' in node.public_ips)
        self.assertTrue('10.0.0.255' in node.private_ips)
        self.assertEqual(node.extra['created_at'], '2015-05-03T15:50:49Z')
        self.assertEqual(node.extra['updated_at'], '2015-05-03T16:00:08Z')
        self.assertEqual(node.extra['billing_cycle'], 'hourly')
        self.assertEqual(node.extra['locked'], False)
        self.assertEqual(node.size.id, 'baremetal_1')
        self.assertEqual(node.size.name, 'Type 1 - 16384 RAM')
        self.assertEqual(node.size.ram, 16384)
        self.assertEqual(node.size.disk, 240)
        self.assertEqual(node.size.price, 0.4)
        self.assertEqual(node.size.extra['line'], 'baremetal')
        self.assertEqual(node.image.id, 'ubuntu_14_04')
        self.assertEqual(node.image.name, 'Ubuntu 14.04 LTS')
        self.assertEqual(node.image.extra['distro'], 'ubuntu')
        self.assertEqual(node.image.extra['version'], '14.04')

    def test_list_nodes_response(self):
        nodes = self.driver.list_nodes('project-id')
        self.assertTrue(isinstance(nodes, list))
        for node in nodes:
            self.assertTrue(isinstance(node, Node))

    def test_list_locations(self):
        locations = self.driver.list_locations()
        self.assertEqual(len(locations), 1)

    def test_list_images(self):
        images = self.driver.list_images()
        self.assertEqual(len(images), 4)

    def test_list_sizes(self):
        sizes = self.driver.list_sizes()
        self.assertEqual(len(sizes), 1)

    def test_create_node(self):
        node = self.driver.create_node(ex_project_id='project-id', name='node-name', size=self.driver.list_sizes()[0], image=self.driver.list_images()[0], location=self.driver.list_locations()[0])
        self.assertTrue(isinstance(node, Node))

    def test_create_node_response(self):
        size = self.driver.list_sizes()[0]
        image = self.driver.list_images()[0]
        location = self.driver.list_locations()[0]
        node = self.driver.create_node(ex_project_id='project-id', name='node-name', image=image, size=size, location=location)
        self.assertTrue(isinstance(node, Node))

    def test_reboot_node(self):
        node = self.driver.list_nodes('project-id')[0]
        self.driver.reboot_node(node)

    def test_reboot_node_response(self):
        node = self.driver.list_nodes('project-id')[0]
        self.driver.reboot_node(node)

    def test_destroy_node(self):
        node = self.driver.list_nodes('project-id')[0]
        self.driver.destroy_node(node)

    def test_destroy_node_response(self):
        node = self.driver.list_nodes('project-id')[0]
        self.driver.destroy_node(node)

    def test_reinstall_node(self):
        node = self.driver.list_nodes('project-id')[0]
        self.driver.ex_reinstall_node(node)

    def test_rescue_node(self):
        node = self.driver.list_nodes('project-id')[0]
        self.driver.ex_rescue_node(node)

    def test_list_key_pairs(self):
        keys = self.driver.list_key_pairs()
        self.assertEqual(len(keys), 3)

    def test_create_key_pair(self):
        key = self.driver.create_key_pair(name='sshkey-name', public_key='ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDI4pIqzpb5g3992h+yr527VRcaB68KE4vPjWPPoiQws49KIs2NMcOzS9QE4641uW1u5ML2HgQdfYKMF/YFGnI1Y6xV637DjhDyZYV9LasUH49npSSJjsBcsk9JGfUpNAOdcgpFzK8V90eiOrOC5YncxdwwG8pwjFI9nNVPCl4hYEu1iXdyysHvkFfS2fklsNjLWrzfafPlaen+qcBxygCA0sFdW/7er50aJeghdBHnE2WhIKLUkJxnKadznfAge7oEe+3LLAPfP+3yHyvp2+H0IzmVfYvAjnzliYetqQ8pg5ZW2BiJzvqz5PebGS70y/ySCNW1qQmJURK/Wc1bt9en root@libcloud')
        self.assertTrue(isinstance(key, KeyPair))

    def test_delete_key_pair(self):
        key = self.driver.list_key_pairs()[0]
        self.driver.delete_key_pair(key)

    def test_ex_list_projects(self):
        projects = self.driver.ex_list_projects()
        self.assertEqual(len(projects), 3)

    def test_ex_get_bgp_config_for_project(self):
        config = self.driver.ex_get_bgp_config_for_project(ex_project_id='4b653fce-6405-4300-9f7d-c587b7888fe5')
        self.assertEqual(config.get('status'), 'enabled')

    def test_ex_get_bgp_config(self):
        config = self.driver.ex_get_bgp_config()
        self.assertEqual(len(config), 2)

    def test_ex_list_nodes_for_project(self):
        nodes = self.driver.ex_list_nodes_for_project(ex_project_id='4b653fce-6405-4300-9f7d-c587b7888fe5')
        self.assertEqual(nodes[0].public_ips, ['147.75.102.193', '2604:1380:2000:c100::3'])

    def test_ex_create_bgp_session(self):
        node = self.driver.list_nodes('project-id')[0]
        session = self.driver.ex_create_bgp_session(node, 'ipv4')
        self.assertEqual(session['status'], 'unknown')

    def test_ex_get_bgp_session(self):
        session = self.driver.ex_get_bgp_session(self.driver.ex_list_bgp_sessions()[0]['id'])
        self.assertEqual(session['status'], 'down')

    def test_ex_list_bgp_sessions_for_project(self):
        sessions = self.driver.ex_list_bgp_sessions_for_project(ex_project_id='4b653fce-6405-4300-9f7d-c587b7888fe5')
        self.assertEqual(sessions['bgp_sessions'][0]['status'], 'down')

    def test_ex_list_bgp_sessions_for_node(self):
        sessions = self.driver.ex_list_bgp_sessions_for_node(self.driver.list_nodes()[0])
        self.assertEqual(sessions['bgp_sessions'][0]['status'], 'down')

    def test_ex_list_bgp_sessions(self):
        sessions = self.driver.ex_list_bgp_sessions()
        self.assertEqual(sessions[0]['status'], 'down')

    def test_ex_delete_bgp_session(self):
        self.driver.ex_delete_bgp_session(session_uuid='08f6b756-758b-4f1f-bfaf-b9b5479822d7')

    def test_ex_list_events_for_node(self):
        events = self.driver.ex_list_events_for_node(self.driver.list_nodes()[0])
        self.assertEqual(events['events'][0]['ip'], '157.52.105.28')

    def test_ex_list_events_for_project(self):
        events = self.driver.ex_list_events_for_project(self.driver.ex_list_projects()[0])
        self.assertEqual(events['meta']['total'], len(events['events']))

    def test_ex_get_node_bandwidth(self):
        node = self.driver.list_nodes('project-id')[0]
        bw = self.driver.ex_get_node_bandwidth(node, 1553194476, 1553198076)
        self.assertTrue(len(bw['bandwidth'][0]['datapoints'][0]) > 0)

    def test_ex_update_node(self):
        node = self.driver.list_nodes('project-id')[0]
        self.driver.ex_update_node(node, description='new_description')

    def test_ex_describe_all_addresses_for_project(self):
        addresses = self.driver.ex_describe_all_addresses_for_project('4b653fce-6405-4300-9f7d-c587b7888fe5')
        self.assertEqual(len(addresses), 5)

    def test_ex_describe_address(self):
        address = self.driver.ex_describe_address(ex_address_id='01c184f5-1413-4b0b-9f6d-ac993f6c9241')
        self.assertEqual(address['network'], '147.75.33.32')

    def test_ex_request_address_reservation(self):
        response = self.driver.ex_request_address_reservation(ex_project_id='3d27fd13-0466-4878-be22-9a4b5595a3df')
        assert response['global_ip']

    def test_ex_associate_address_with_node(self):
        node = self.driver.list_nodes('project-id')[0]
        response = self.driver.ex_associate_address_with_node(node, '147.75.40.2/32')
        assert response['enabled']

    def test_ex_disassociate_address_with_node(self):
        node = self.driver.list_nodes('project-id')[0]
        assignments = self.driver.ex_list_ip_assignments_for_node(node)
        for ip_assignment in assignments['ip_addresses']:
            if ip_assignment['gateway'] == '147.75.40.2':
                self.driver.ex_disassociate_address(ip_assignment['id'])
                break

    def test_list_volumes(self):
        volumes = self.driver.list_volumes()
        assert len(volumes) == 2
        assert len(volumes[0].extra['attachments']) == 0

    def test_create_volume(self):
        location = self.driver.list_locations()[0]
        volume = self.driver.create_volume(10, location, description='test volume', plan='storage_1', ex_project_id='3d27fd13-0466-4878-be22-9a4b5595a3df')
        assert len(volume.extra['attachments']) == 0
        assert not volume.extra['locked']

    def test_attach_volume(self):
        attached = False
        volumes = self.driver.ex_list_volumes_for_project(ex_project_id='3d27fd13-0466-4878-be22-9a4b5595a3df')
        node = self.driver.ex_list_nodes_for_project(ex_project_id='3d27fd13-0466-4878-be22-9a4b5595a3df')[0]
        for vol in volumes:
            if len(vol.extra['attachments']) == 0:
                attached = self.driver.attach_volume(node, vol)
                break
        assert attached

    def test_detach_volume(self):
        detached = False
        volumes = self.driver.ex_list_volumes_for_project(ex_project_id='3d27fd13-0466-4878-be22-9a4b5595a3df')
        for vol in volumes:
            if len(vol.extra['attachments']) > 0:
                detached = self.driver.detach_volume(vol)
                break
        assert detached

    def test_destroy_volume(self):
        destroyed = False
        volumes = self.driver.ex_list_volumes_for_project(ex_project_id='3d27fd13-0466-4878-be22-9a4b5595a3df')
        for vol in volumes:
            if len(vol.extra['attachments']) == 0:
                destroyed = self.driver.destroy_volume(vol)
                break
        assert destroyed