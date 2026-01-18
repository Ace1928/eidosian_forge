from unittest import mock
import testtools
from heatclient.common import event_utils
from heatclient.v1 import events as hc_ev
from heatclient.v1 import resources as hc_res
@staticmethod
def _mock_resource(resource_id, nested_id=None):
    res_info = {'links': [{'href': 'http://heat/foo', 'rel': 'self'}, {'href': 'http://heat/foo2', 'rel': 'resource'}], 'logical_resource_id': resource_id, 'physical_resource_id': resource_id, 'resource_status': 'CREATE_COMPLETE', 'resource_status_reason': 'state changed', 'resource_type': 'OS::Nested::Server', 'updated_time': '2014-01-06T16:14:26Z'}
    if nested_id:
        nested_link = {'href': 'http://heat/%s' % nested_id, 'rel': 'nested'}
        res_info['links'].append(nested_link)
    return hc_res.Resource(manager=None, info=res_info)