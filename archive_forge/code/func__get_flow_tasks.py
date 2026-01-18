from unittest import mock
import futurist
import glance_store as store
from oslo_config import cfg
from taskflow.patterns import linear_flow
import glance.async_
from glance.async_.flows import api_image_import
import glance.tests.utils as test_utils
def _get_flow_tasks(self, flow):
    flow_comp = []
    for c, p in flow.iter_nodes():
        if isinstance(c, linear_flow.Flow):
            flow_comp += self._get_flow_tasks(c)
        else:
            name = str(c).split('-')
            if len(name) > 1:
                flow_comp.append(name[1])
    return flow_comp