import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_RANCHER
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.rancher import RancherContainerDriver
class RancherContainerDriverTestCase(unittest.TestCase):

    def setUp(self):
        self.driver = RancherContainerDriver(*CONTAINER_PARAMS_RANCHER)

    def test_ex_list_stacks(self):
        stacks = self.driver.ex_list_stacks()
        self.assertEqual(len(stacks), 6)
        self.assertEqual(stacks[0]['id'], '1e1')

    def test_ex_deploy_stack(self):
        stack = self.driver.ex_deploy_stack(name='newstack', environment={'root_password': 'password'})
        self.assertEqual(stack['id'], '1e9')
        self.assertEqual(stack['environment']['root_password'], 'password')

    def test_ex_get_stack(self):
        stack = self.driver.ex_get_stack('1e9')
        self.assertEqual(stack['id'], '1e9')
        self.assertEqual(stack['environment']['root_password'], 'password')

    def test_ex_search_stacks(self):
        stacks = self.driver.ex_search_stacks({'healthState': 'healthy'})
        self.assertEqual(len(stacks), 6)
        self.assertEqual(stacks[0]['healthState'], 'healthy')

    def test_ex_destroy_stack(self):
        response = self.driver.ex_destroy_stack('1e10')
        self.assertEqual(response, True)

    def test_ex_activate_stack(self):
        response = self.driver.ex_activate_stack('1e1')
        self.assertEqual(response, True)

    def test_ex_deactivate_stack(self):
        response = self.driver.ex_activate_stack('1e1')
        self.assertEqual(response, True)

    def test_ex_list_services(self):
        services = self.driver.ex_list_services()
        self.assertEqual(len(services), 4)
        self.assertEqual(services[0]['id'], '1s1')

    def test_ex_deploy_service(self):
        image = ContainerImage(id='hastebin', name='hastebin', path='rlister/hastebin', version='latest', driver=None)
        service = self.driver.ex_deploy_service(name='newservice', environment_id='1e1', image=image, environment={'root_password': 'password'})
        self.assertEqual(service['id'], '1s13')
        self.assertEqual(service['environmentId'], '1e6')
        self.assertEqual(service['launchConfig']['environment']['root_password'], 'password')
        self.assertEqual(service['launchConfig']['imageUuid'], 'docker:rlister/hastebin:latest')

    def test_ex_get_service(self):
        service = self.driver.ex_get_service('1s13')
        self.assertEqual(service['id'], '1s13')
        self.assertEqual(service['environmentId'], '1e6')
        self.assertEqual(service['launchConfig']['environment']['root_password'], 'password')

    def test_ex_search_services(self):
        services = self.driver.ex_search_services({'healthState': 'healthy'})
        self.assertEqual(len(services), 2)
        self.assertEqual(services[0]['healthState'], 'healthy')

    def test_ex_destroy_service(self):
        response = self.driver.ex_destroy_service('1s13')
        self.assertEqual(response, True)

    def test_ex_activate_service(self):
        response = self.driver.ex_activate_service('1s6')
        self.assertEqual(response, True)

    def test_ex_deactivate_service(self):
        response = self.driver.ex_activate_service('1s6')
        self.assertEqual(response, True)

    def test_list_containers(self):
        containers = self.driver.list_containers()
        self.assertEqual(len(containers), 2)
        self.assertEqual(containers[0].id, '1i1')

    def test_deploy_container(self):
        container = self.driver.deploy_container(name='newcontainer', image=ContainerImage(id='hastebin', name='hastebin', path='rlister/hastebin', version='latest', driver=None), environment={'STORAGE_TYPE': 'file'}, networkMode='managed')
        self.assertEqual(container.id, '1i31')
        self.assertEqual(container.name, 'newcontainer')
        self.assertEqual(container.extra['environment'], {'STORAGE_TYPE': 'file'})

    def test_get_container(self):
        container = self.driver.get_container('1i31')
        self.assertEqual(container.id, '1i31')
        self.assertEqual(container.name, 'newcontainer')
        self.assertEqual(container.extra['environment'], {'STORAGE_TYPE': 'file'})

    def test_start_container(self):
        container = self.driver.get_container('1i31')
        started = container.start()
        self.assertEqual(started.id, '1i31')
        self.assertEqual(started.name, 'newcontainer')
        self.assertEqual(started.state, 'pending')
        self.assertEqual(started.extra['state'], 'starting')

    def test_stop_container(self):
        container = self.driver.get_container('1i31')
        stopped = container.stop()
        self.assertEqual(stopped.id, '1i31')
        self.assertEqual(stopped.name, 'newcontainer')
        self.assertEqual(stopped.state, 'pending')
        self.assertEqual(stopped.extra['state'], 'stopping')

    def test_ex_search_containers(self):
        containers = self.driver.ex_search_containers({'state': 'running'})
        self.assertEqual(len(containers), 1)

    def test_destroy_container(self):
        container = self.driver.get_container('1i31')
        destroyed = container.destroy()
        self.assertEqual(destroyed.id, '1i31')
        self.assertEqual(destroyed.name, 'newcontainer')
        self.assertEqual(destroyed.state, 'pending')
        self.assertEqual(destroyed.extra['state'], 'stopping')