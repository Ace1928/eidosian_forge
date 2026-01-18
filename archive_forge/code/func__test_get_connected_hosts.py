from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def _test_get_connected_hosts(self, in_maintenance_mode, m1_accessible=True):
    session = mock.Mock()
    ds_ref = vim_util.get_moref('ds-0', 'Datastore')
    ds = datastore.Datastore(ds_ref, 'ds-name')
    ds.get_summary = mock.Mock()
    ds.get_summary.return_value.accessible = False
    self.assertEqual([], ds.get_connected_hosts(session))
    ds.get_summary.return_value.accessible = True
    m1 = HostMount('m1', MountInfo('readWrite', True, m1_accessible))
    m2 = HostMount('m2', MountInfo('read', True, True))
    m3 = HostMount('m3', MountInfo('readWrite', False, True))
    m4 = HostMount('m4', MountInfo('readWrite', True, False))
    ds.get_summary.assert_called_once_with(session)

    class Prop(object):
        DatastoreHostMount = [m1, m2, m3, m4]

    class HostRuntime(object):
        inMaintenanceMode = in_maintenance_mode

    class HostProp(object):
        name = 'runtime'
        val = HostRuntime()

    class Object(object):
        obj = 'm1'
        propSet = [HostProp()]

    class Runtime(object):
        objects = [Object()]
    session.invoke_api = mock.Mock(side_effect=[Prop(), Runtime()])
    hosts = ds.get_connected_hosts(session)
    calls = [mock.call(vim_util, 'get_object_property', session.vim, ds_ref, 'host')]
    if m1_accessible:
        calls.append(mock.call(vim_util, 'get_properties_for_a_collection_of_objects', session.vim, 'HostSystem', ['m1'], ['runtime']))
    self.assertEqual(calls, session.invoke_api.mock_calls)
    return hosts