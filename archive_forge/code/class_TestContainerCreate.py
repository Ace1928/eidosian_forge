import copy
from requests_mock.contrib import fixture
from openstackclient.object.v1 import container as container_cmds
from openstackclient.tests.unit.object.v1 import fakes as object_fakes
class TestContainerCreate(TestContainerAll):
    columns = ('account', 'container', 'x-trans-id')

    def setUp(self):
        super(TestContainerCreate, self).setUp()
        self.cmd = container_cmds.CreateContainer(self.app, None)

    def test_object_create_container_single(self):
        self.requests_mock.register_uri('PUT', object_fakes.ENDPOINT + '/ernie', headers={'x-trans-id': '314159'}, status_code=200)
        arglist = ['ernie']
        verifylist = [('containers', ['ernie'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        datalist = [(object_fakes.ACCOUNT_ID, 'ernie', '314159')]
        self.assertEqual(datalist, list(data))

    def test_object_create_container_storage_policy(self):
        self.requests_mock.register_uri('PUT', object_fakes.ENDPOINT + '/ernie', headers={'x-trans-id': '314159', 'x-storage-policy': 'o1--sr-r3'}, status_code=200)
        arglist = ['ernie', '--storage-policy', 'o1--sr-r3']
        verifylist = [('containers', ['ernie']), ('storage_policy', 'o1--sr-r3')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        datalist = [(object_fakes.ACCOUNT_ID, 'ernie', '314159')]
        self.assertEqual(datalist, list(data))

    def test_object_create_container_public(self):
        self.requests_mock.register_uri('PUT', object_fakes.ENDPOINT + '/ernie', headers={'x-trans-id': '314159', 'x-container-read': '.r:*,.rlistings'}, status_code=200)
        arglist = ['ernie', '--public']
        verifylist = [('containers', ['ernie']), ('public', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        datalist = [(object_fakes.ACCOUNT_ID, 'ernie', '314159')]
        self.assertEqual(datalist, list(data))

    def test_object_create_container_more(self):
        self.requests_mock.register_uri('PUT', object_fakes.ENDPOINT + '/ernie', headers={'x-trans-id': '314159'}, status_code=200)
        self.requests_mock.register_uri('PUT', object_fakes.ENDPOINT + '/bert', headers={'x-trans-id': '42'}, status_code=200)
        arglist = ['ernie', 'bert']
        verifylist = [('containers', ['ernie', 'bert'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        datalist = [(object_fakes.ACCOUNT_ID, 'ernie', '314159'), (object_fakes.ACCOUNT_ID, 'bert', '42')]
        self.assertEqual(datalist, list(data))