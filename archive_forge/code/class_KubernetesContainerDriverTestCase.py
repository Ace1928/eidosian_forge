import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_KUBERNETES
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.test.common.test_kubernetes import KubernetesAuthTestCaseMixin
from libcloud.container.drivers.kubernetes import (
class KubernetesContainerDriverTestCase(unittest.TestCase, KubernetesAuthTestCaseMixin):
    driver_cls = KubernetesContainerDriver

    def setUp(self):
        KubernetesContainerDriver.connectionCls.conn_class = KubernetesMockHttp
        KubernetesMockHttp.type = None
        KubernetesMockHttp.use_param = 'a'
        self.driver = KubernetesContainerDriver(*CONTAINER_PARAMS_KUBERNETES)

    def test_list_containers(self):
        containers = self.driver.list_containers()
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].id, 'docker://3c48b5cda79bce4c8866f02a3b96a024edb8f660d10e7d1755e9ced49ef47b36')
        self.assertEqual(containers[0].name, 'hello-world')

    def test_deploy_container(self):
        image = ContainerImage(id=None, name='hello-world', path=None, driver=self.driver, version=None)
        container = self.driver.deploy_container('hello-world', image=image)
        self.assertEqual(container.name, 'hello-world')

    def test_get_container(self):
        container = self.driver.get_container('docker://3c48b5cda79bce4c8866f02a3b96a024edb8f660d10e7d1755e9ced49ef47b36')
        assert container.id == 'docker://3c48b5cda79bce4c8866f02a3b96a024edb8f660d10e7d1755e9ced49ef47b36'

    def test_list_namespaces(self):
        namespaces = self.driver.list_namespaces()
        self.assertEqual(len(namespaces), 2)
        self.assertEqual(namespaces[0].id, 'default')
        self.assertEqual(namespaces[0].name, 'default')

    def test_get_namespace(self):
        namespace = self.driver.get_namespace('default')
        self.assertEqual(namespace.id, 'default')
        self.assertEqual(namespace.name, 'default')

    def test_create_namespace(self):
        namespace = self.driver.create_namespace('test')
        self.assertEqual(namespace.id, 'test')
        self.assertEqual(namespace.name, 'test')

    def test_delete_namespace(self):
        namespace = self.driver.get_namespace('default')
        result = self.driver.delete_namespace(namespace)
        self.assertTrue(result)

    def test_list_pods(self):
        pods = self.driver.ex_list_pods()
        self.assertEqual(len(pods), 1)
        self.assertEqual(pods[0].id, '1fad5411-b9af-11e5-8701-0050568157ec')
        self.assertEqual(pods[0].name, 'hello-world')

    def test_destroy_pod(self):
        result = self.driver.ex_destroy_pod('default', 'default')
        self.assertTrue(result)

    def test_list_nodes(self):
        nodes = self.driver.ex_list_nodes()
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].id, '45949cbb-b99d-11e5-8d53-0050568157ec')
        self.assertEqual(nodes[0].name, '127.0.0.1')

    def test_destroy_node(self):
        result = self.driver.ex_destroy_node('127.0.0.1')
        self.assertTrue(result)

    def test_get_version(self):
        version = self.driver.ex_get_version()
        self.assertEqual(version, 'v1.20.8-gke.900')

    def test_list_nodes_metrics(self):
        nodes_metrics = self.driver.ex_list_nodes_metrics()
        self.assertEqual(len(nodes_metrics), 1)
        self.assertEqual(nodes_metrics[0]['metadata']['name'], 'gke-cluster-3-default-pool-76fd57f7-l83v')

    def test_list_pods_metrics(self):
        pods_metrics = self.driver.ex_list_pods_metrics()
        self.assertEqual(len(pods_metrics), 10)
        self.assertEqual(pods_metrics[0]['metadata']['name'], 'gke-metrics-agent-sfjzj')
        self.assertEqual(pods_metrics[1]['metadata']['name'], 'stackdriver-metadata-agent-cluster-level-849ff68b6d-fphxl')
        self.assertEqual(pods_metrics[2]['metadata']['name'], 'event-exporter-gke-67986489c8-g47rz')

    def test_list_services(self):
        services = self.driver.ex_list_services()
        self.assertEqual(len(services), 4)
        self.assertEqual(services[0]['metadata']['name'], 'kubernetes')
        self.assertEqual(services[1]['metadata']['name'], 'default-http-backend')
        self.assertEqual(services[2]['metadata']['name'], 'kube-dns')

    def test_list_deployments(self):
        deployments = self.driver.ex_list_deployments()
        self.assertEqual(len(deployments), 7)
        deployment = deployments[0]
        self.assertEqual(deployment.id, 'aea45586-9a4a-4a01-805c-719f431316c0')
        self.assertEqual(deployment.name, 'event-exporter-gke')
        self.assertEqual(deployment.namespace, 'kube-system')
        for deployment in deployments:
            self.assertIsInstance(deployment.replicas, int)
            self.assertIsInstance(deployment.selector, dict)

    def test_to_n_bytes(self):
        memory = '0'
        self.assertEqual(to_n_bytes(memory), 0)
        memory = '1000Ki'
        self.assertEqual(to_n_bytes(memory), 1024000)
        memory = '100K'
        self.assertEqual(to_n_bytes(memory), 100000)
        memory = '512Mi'
        self.assertEqual(to_n_bytes(memory), 536870912)
        memory = '900M'
        self.assertEqual(to_n_bytes(memory), 900000000)
        memory = '10Gi'
        self.assertEqual(to_n_bytes(memory), 10737418240)
        memory = '10G'
        self.assertEqual(to_n_bytes(memory), 10000000000)

    def test_to_memory_str(self):
        memory = 0
        self.assertEqual(to_memory_str(memory), '0K')
        memory = 1024000
        self.assertEqual(to_memory_str(memory), '1000Ki')
        memory = 100000
        self.assertEqual(to_memory_str(memory), '100K')
        memory = 536870912
        self.assertEqual(to_memory_str(memory), '512Mi')
        memory = 900000000
        self.assertEqual(to_memory_str(memory), '900M')
        memory = 10737418240
        self.assertEqual(to_memory_str(memory), '10Gi')
        memory = 10000000000
        self.assertEqual(to_memory_str(memory), '10G')

    def test_to_cpu_str(self):
        cpu = 0
        self.assertEqual(to_cpu_str(cpu), '0')
        cpu = 0.5
        self.assertEqual(to_cpu_str(cpu), '500m')
        cpu = 2
        self.assertEqual(to_cpu_str(cpu), '2000m')
        cpu = 1e-06
        self.assertEqual(to_cpu_str(cpu), '1u')
        cpu = 0.0005
        self.assertEqual(to_cpu_str(cpu), '500u')
        cpu = 1e-09
        self.assertEqual(to_cpu_str(cpu), '1n')
        cpu = 5e-07
        self.assertEqual(to_cpu_str(cpu), '500n')

    def test_to_n_cpus(self):
        cpu = '0m'
        self.assertEqual(to_n_cpus(cpu), 0)
        cpu = '2'
        self.assertEqual(to_n_cpus(cpu), 2)
        cpu = '500m'
        self.assertEqual(to_n_cpus(cpu), 0.5)
        cpu = '500m'
        self.assertEqual(to_n_cpus(cpu), 0.5)
        cpu = '2000m'
        self.assertEqual(to_n_cpus(cpu), 2)
        cpu = '1u'
        self.assertEqual(to_n_cpus(cpu), 1e-06)
        cpu = '500u'
        self.assertEqual(to_n_cpus(cpu), 0.0005)
        cpu = '1n'
        self.assertEqual(to_n_cpus(cpu), 1e-09)
        cpu = '500n'
        self.assertEqual(to_n_cpus(cpu), 5e-07)

    def test_sum_resources(self):
        resource_1 = {'cpu': '1', 'memory': '1000Mi'}
        resource_2 = {'cpu': '2', 'memory': '2000Mi'}
        self.assertDictEqual(sum_resources(resource_1, resource_2), {'cpu': '3000m', 'memory': '3000Mi'})
        resource_3 = {'cpu': '1500m', 'memory': '1Gi'}
        self.assertDictEqual(sum_resources(resource_1, resource_2, resource_3), {'cpu': '4500m', 'memory': '4024Mi'})