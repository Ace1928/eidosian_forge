from unittest import mock
import testtools
from heatclient.common import event_utils
from heatclient.v1 import events as hc_ev
from heatclient.v1 import resources as hc_res
@staticmethod
def _mock_event(event_id, resource_id, resource_status='CREATE_COMPLETE'):
    ev_info = {'links': [{'href': 'http://heat/foo', 'rel': 'self'}, {'href': 'http://heat/stacks/astack', 'rel': 'stack'}], 'logical_resource_id': resource_id, 'physical_resource_id': resource_id, 'resource_name': resource_id, 'resource_status': resource_status, 'resource_status_reason': 'state changed', 'event_time': '2014-12-05T14:14:30Z', 'id': event_id}
    return hc_ev.Event(manager=None, info=ev_info)