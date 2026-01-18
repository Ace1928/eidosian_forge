import logging
import os
from oslo_config import cfg
from osprofiler.drivers import base
from osprofiler import initializer
from osprofiler import opts
from osprofiler import profiler
from osprofiler.tests import test
def _assert_child_dict(self, child, base_id, parent_id, name, fn_name):
    self.assertEqual(parent_id, child['parent_id'])
    exp_info = {'name': 'rpc', 'service': self.SERVICE, 'project': self.PROJECT}
    self._assert_dict(child['info'], **exp_info)
    raw_start = child['info']['meta.raw_payload.%s-start' % name]
    self.assertEqual(fn_name, raw_start['info']['function']['name'])
    exp_raw = {'name': '%s-start' % name, 'service': self.SERVICE, 'trace_id': child['trace_id'], 'project': self.PROJECT, 'base_id': base_id}
    self._assert_dict(raw_start, **exp_raw)
    raw_stop = child['info']['meta.raw_payload.%s-stop' % name]
    exp_raw['name'] = '%s-stop' % name
    self._assert_dict(raw_stop, **exp_raw)