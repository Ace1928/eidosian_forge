import argparse
from unittest import mock
from openstack import exceptions
from openstack.identity.v3 import project
import testtools
from osc_lib.cli import identity as cli_identity
from osc_lib.tests import utils as test_utils
class IdentityUtilsTestCase(test_utils.TestCase):

    def test_add_project_owner_option_to_parser(self):
        parser = argparse.ArgumentParser()
        cli_identity.add_project_owner_option_to_parser(parser)
        parsed_args = parser.parse_args(['--project', 'project1', '--project-domain', 'domain1'])
        self.assertEqual('project1', parsed_args.project)
        self.assertEqual('domain1', parsed_args.project_domain)

    def test_find_project(self):
        sdk_connection = mock.Mock()
        sdk_find_project = sdk_connection.identity.find_project
        sdk_find_project.return_value = mock.sentinel.project1
        ret = cli_identity.find_project(sdk_connection, 'project1')
        self.assertEqual(mock.sentinel.project1, ret)
        sdk_find_project.assert_called_once_with('project1', ignore_missing=False, domain_id=None)

    def test_find_project_with_domain(self):
        domain1 = mock.Mock()
        domain1.id = 'id-domain1'
        sdk_connection = mock.Mock()
        sdk_find_domain = sdk_connection.identity.find_domain
        sdk_find_domain.return_value = domain1
        sdk_find_project = sdk_connection.identity.find_project
        sdk_find_project.return_value = mock.sentinel.project1
        ret = cli_identity.find_project(sdk_connection, 'project1', 'domain1')
        self.assertEqual(mock.sentinel.project1, ret)
        sdk_find_domain.assert_called_once_with('domain1', ignore_missing=False)
        sdk_find_project.assert_called_once_with('project1', ignore_missing=False, domain_id='id-domain1')

    def test_find_project_with_forbidden_exception(self):
        sdk_connection = mock.Mock()
        sdk_find_project = sdk_connection.identity.find_project
        exc = exceptions.HttpException()
        exc.status_code = 403
        sdk_find_project.side_effect = exc
        ret = cli_identity.find_project(sdk_connection, 'project1')
        self.assertIsInstance(ret, project.Project)
        self.assertEqual('project1', ret.id)
        self.assertEqual('project1', ret.name)

    def test_find_project_with_generic_exception(self):
        sdk_connection = mock.Mock()
        sdk_find_project = sdk_connection.identity.find_project
        exc = exceptions.HttpException()
        exc.status_code = 499
        sdk_find_project.side_effect = exc
        with testtools.ExpectedException(exceptions.HttpException):
            cli_identity.find_project(sdk_connection, 'project1')