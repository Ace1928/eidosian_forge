import copy
from unittest import mock
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_group
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class ResourceGroupSkiplistTest(common.HeatTestCase):
    """This class tests ResourceGroup._name_skiplist()."""
    scenarios = [('1', dict(data_in=None, rm_list=[], nested_rsrcs=[], expected=[], saved=False, fallback=False, rm_mode='append')), ('2', dict(data_in='0,1,2', rm_list=[], nested_rsrcs=[], expected=['0', '1', '2'], saved=False, fallback=False, rm_mode='append')), ('3', dict(data_in='1,3', rm_list=['6'], nested_rsrcs=['0', '1', '3'], expected=['1', '3'], saved=False, fallback=False, rm_mode='append')), ('4', dict(data_in='0,1', rm_list=['id-7'], nested_rsrcs=['0', '1', '3'], expected=['0', '1'], saved=False, fallback=False, rm_mode='append')), ('5', dict(data_in='0,1', rm_list=['3'], nested_rsrcs=['0', '1', '3'], expected=['0', '1', '3'], saved=True, fallback=False, rm_mode='append')), ('6', dict(data_in='0,1', rm_list=['id-3'], nested_rsrcs=['0', '1', '3'], expected=['0', '1', '3'], saved=True, fallback=False, rm_mode='append')), ('7', dict(data_in='0,1', rm_list=['id-3'], nested_rsrcs=['0', '1', '3'], expected=['3'], saved=True, fallback=False, rm_mode='update')), ('8', dict(data_in='1', rm_list=[], nested_rsrcs=['0', '1', '2'], expected=[], saved=True, fallback=False, rm_mode='update')), ('9', dict(data_in='0,1', rm_list=['id-3'], nested_rsrcs=['0', '1', '3'], expected=['0', '1', '3'], saved=True, fallback=True, rm_mode='append')), ('A', dict(data_in='0,1', rm_list=['id-3'], nested_rsrcs=['0', '1', '3'], expected=['3'], saved=True, fallback=True, rm_mode='update'))]

    def test_skiplist(self):
        stack = utils.parse_stack(template)
        resg = stack['group1']
        if self.data_in is not None:
            resg.resource_id = 'foo'
        properties = mock.MagicMock()
        p_data = {'removal_policies': [{'resource_list': self.rm_list}], 'removal_policies_mode': self.rm_mode}
        properties.get.side_effect = p_data.get
        resg.data = mock.Mock()
        resg.data.return_value.get.return_value = self.data_in
        resg.data_set = mock.Mock()
        mock_inspect = mock.Mock()
        self.patchobject(grouputils.GroupInspector, 'from_parent_resource', return_value=mock_inspect)
        mock_inspect.member_names.return_value = self.nested_rsrcs
        if not self.fallback:
            refs_map = {n: 'id-%s' % n for n in self.nested_rsrcs}
            resg.get_output = mock.Mock(return_value=refs_map)
        else:
            resg.get_output = mock.Mock(side_effect=exception.NotFound)

            def stack_contains(name):
                return name in self.nested_rsrcs

            def by_refid(name):
                rid = name.replace('id-', '')
                if rid not in self.nested_rsrcs:
                    return None
                res = mock.Mock()
                res.name = rid
                return res
            nested = mock.MagicMock()
            nested.__contains__.side_effect = stack_contains
            nested.__iter__.side_effect = iter(self.nested_rsrcs)
            nested.resource_by_refid.side_effect = by_refid
            resg.nested = mock.Mock(return_value=nested)
        resg._update_name_skiplist(properties)
        if self.saved:
            resg.data_set.assert_called_once_with('name_blacklist', ','.join(self.expected))
        else:
            resg.data_set.assert_not_called()
            self.assertEqual(set(self.expected), resg._name_skiplist())