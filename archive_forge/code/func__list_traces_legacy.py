from urllib import parse as parser
from debtcollector import removals
from oslo_config import cfg
from oslo_serialization import jsonutils
from osprofiler.drivers import base
from osprofiler import exc
def _list_traces_legacy(self, fields):
    ids = self.db.scan_iter(match=self.namespace + '*')
    traces = [jsonutils.loads(self.db.get(i)) for i in ids]
    traces.sort(key=lambda x: x['timestamp'])
    seen_ids = set()
    result = []
    for trace in traces:
        if trace['base_id'] not in seen_ids:
            seen_ids.add(trace['base_id'])
            result.append({key: value for key, value in trace.items() if key in fields})
    return result