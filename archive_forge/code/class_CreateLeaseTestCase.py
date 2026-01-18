import argparse
from datetime import datetime
from unittest import mock
from blazarclient import exception
from blazarclient import shell
from blazarclient import tests
from blazarclient.v1.shell_commands import leases
@mock.patch('blazarclient.v1.shell_commands.leases._utc_now', mock_time)
class CreateLeaseTestCase(tests.TestCase):

    def setUp(self):
        super(CreateLeaseTestCase, self).setUp()
        self.cl = leases.CreateLease(shell.BlazarShell(), mock.Mock())

    def test_args2body_correct_phys_res_params(self):
        args = argparse.Namespace(start='2020-07-24 20:00', end='2020-08-09 22:30', before_end='2020-08-09 21:30', events=[], name='lease-test', reservations=[], physical_reservations=['min=1,max=2,hypervisor_properties=["and", [">=", "$vcpus", "2"], [">=", "$memory_mb", "2048"]],resource_properties=["==", "$extra_key", "extra_value"],before_end=default'])
        expected = {'start': '2020-07-24 20:00', 'end': '2020-08-09 22:30', 'before_end': '2020-08-09 21:30', 'events': [], 'name': 'lease-test', 'reservations': [{'min': 1, 'max': 2, 'hypervisor_properties': '["and", [">=", "$vcpus", "2"], [">=", "$memory_mb", "2048"]]', 'resource_properties': '["==", "$extra_key", "extra_value"]', 'resource_type': 'physical:host', 'before_end': 'default'}]}
        self.assertDictEqual(self.cl.args2body(args), expected)

    def test_args2body_incorrect_phys_res_params(self):
        args = argparse.Namespace(start='2020-07-24 20:00', end='2020-08-09 22:30', before_end='2020-08-09 21:30', events=[], name='lease-test', reservations=[], physical_reservations=['incorrect_param=1,min=1,max=2,hypervisor_properties=["and", [">=", "$vcpus", "2"], [">=", "$memory_mb", "2048"]],resource_properties=["==", "$extra_key", "extra_value"]'])
        self.assertRaises(exception.IncorrectLease, self.cl.args2body, args)

    def test_args2body_duplicated_phys_res_params(self):
        args = argparse.Namespace(start='2020-07-24 20:00', end='2020-08-09 22:30', before_end='2020-08-09 21:30', events=[], name='lease-test', reservations=[], physical_reservations=['min=1,min=1,max=2,hypervisor_properties=["and", [">=", "$vcpus", "2"], [">=", "$memory_mb", "2048"]],resource_properties=["==", "$extra_key", "extra_value"]'])
        self.assertRaises(exception.DuplicatedLeaseParameters, self.cl.args2body, args)

    def test_args2body_correct_instance_res_params(self):
        args = argparse.Namespace(start='2020-07-24 20:00', end='2020-08-09 22:30', before_end='2020-08-09 21:30', events=[], name='lease-test', reservations=['vcpus=4,memory_mb=1024,disk_gb=10,amount=2,affinity=True,resource_properties=["==", "$extra_key", "extra_value"],resource_type=virtual:instance'], physical_reservations=['min=1,max=2,hypervisor_properties=["and", [">=", "$vcpus", "2"], [">=", "$memory_mb", "2048"]],resource_properties=["==", "$extra_key", "extra_value"],before_end=default'])
        expected = {'start': '2020-07-24 20:00', 'end': '2020-08-09 22:30', 'before_end': '2020-08-09 21:30', 'events': [], 'name': 'lease-test', 'reservations': [{'min': 1, 'max': 2, 'hypervisor_properties': '["and", [">=", "$vcpus", "2"], [">=", "$memory_mb", "2048"]]', 'resource_properties': '["==", "$extra_key", "extra_value"]', 'resource_type': 'physical:host', 'before_end': 'default'}, {'vcpus': 4, 'memory_mb': 1024, 'disk_gb': 10, 'amount': 2, 'affinity': 'True', 'resource_properties': '["==", "$extra_key", "extra_value"]', 'resource_type': 'virtual:instance'}]}
        self.assertDictEqual(self.cl.args2body(args), expected)

    def test_args2body_start_now(self):
        args = argparse.Namespace(start='now', end='2030-08-09 22:30', before_end='2030-08-09 21:30', events=[], name='lease-test', reservations=[], physical_reservations=['min=1,max=2,hypervisor_properties=["and", [">=", "$vcpus", "2"], [">=", "$memory_mb", "2048"]],resource_properties=["==", "$extra_key", "extra_value"],before_end=default'])
        expected = {'start': 'now', 'end': '2030-08-09 22:30', 'before_end': '2030-08-09 21:30', 'events': [], 'name': 'lease-test', 'reservations': [{'min': 1, 'max': 2, 'hypervisor_properties': '["and", [">=", "$vcpus", "2"], [">=", "$memory_mb", "2048"]]', 'resource_properties': '["==", "$extra_key", "extra_value"]', 'resource_type': 'physical:host', 'before_end': 'default'}]}
        self.assertDictEqual(self.cl.args2body(args), expected)