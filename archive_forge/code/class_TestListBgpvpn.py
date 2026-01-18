import copy
import operator
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient.osc.v2.networking_bgpvpn import bgpvpn
from neutronclient.tests.unit.osc.v2.networking_bgpvpn import fakes
class TestListBgpvpn(fakes.TestNeutronClientBgpvpn):

    def setUp(self):
        super(TestListBgpvpn, self).setUp()
        self.cmd = bgpvpn.ListBgpvpn(self.app, self.namespace)

    def test_list_all_bgpvpn(self):
        count = 3
        fake_bgpvpns = fakes.create_bgpvpns(count=count)
        self.networkclient.bgpvpns = mock.Mock(return_value=fake_bgpvpns)
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.networkclient.bgpvpns.assert_called_once()
        self.assertEqual(headers, list(headers_short))
        self.assertListItemEqual(list(data), [_get_data(fake_bgpvpn, columns_short) for fake_bgpvpn in fake_bgpvpns])

    def test_list_all_bgpvpn_long_mode(self):
        count = 3
        fake_bgpvpns = fakes.create_bgpvpns(count=count)
        self.networkclient.bgpvpns = mock.Mock(return_value=fake_bgpvpns)
        arglist = ['--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.networkclient.bgpvpns.assert_called_once()
        self.assertEqual(headers, list(headers_long))
        self.assertListItemEqual(list(data), [_get_data(fake_bgpvpn, columns_long) for fake_bgpvpn in fake_bgpvpns])

    def test_list_project_bgpvpn(self):
        count = 3
        project_id = 'list_fake_project_id'
        attrs = {'tenant_id': project_id}
        fake_bgpvpns = fakes.create_bgpvpns(count=count, attrs=attrs)
        self.networkclient.bgpvpns = mock.Mock(return_value=fake_bgpvpns)
        arglist = ['--project', project_id]
        verifylist = [('project', project_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.networkclient.bgpvpns.assert_called_once_with(tenant_id=project_id)
        self.assertEqual(headers, list(headers_short))
        self.assertListItemEqual(list(data), [_get_data(fake_bgpvpn, columns_short) for fake_bgpvpn in fake_bgpvpns])

    def test_list_bgpvpn_with_filters(self):
        count = 3
        name = 'fake_id0'
        layer_type = 'l2'
        attrs = {'type': layer_type}
        fake_bgpvpns = fakes.create_bgpvpns(count=count, attrs=attrs)
        returned_bgpvpn = fake_bgpvpns[0]
        self.networkclient.bgpvpns = mock.Mock(return_value=[returned_bgpvpn])
        arglist = ['--property', 'name=%s' % name, '--property', 'type=%s' % layer_type]
        verifylist = [('property', {'name': name, 'type': layer_type})]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.networkclient.bgpvpns.assert_called_once_with(name=name, type=layer_type)
        self.assertEqual(headers, list(headers_short))
        self.assertListItemEqual(list(data), [_get_data(returned_bgpvpn, columns_short)])