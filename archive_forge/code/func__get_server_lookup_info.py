import importlib
import importlib.metadata
import typing as t
import traceback
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
def _get_server_lookup_info(self, default_realm: str) -> t.Dict[str, t.Any]:
    if not HAS_DNSRESOLVER:
        return {}
    records: t.List[t.Dict[str, t.Any]] = []
    res: t.Dict[str, t.Any] = {'exception': None, 'default_server': None, 'default_port': None, 'records': records}
    try:
        srv_record = f'_ldap._tcp.dc._msdcs.{default_realm}'
        for rec in dns.resolver.resolve(srv_record, 'SRV'):
            records.append({'target': str(rec.target), 'port': rec.port, 'weight': rec.weight, 'priority': rec.priority})
        highest_record = next(iter(sorted(records, key=lambda k: (k['priority'], -k['weight']))), None)
        if highest_record:
            res['default_server'] = highest_record['target'].rstrip('.')
            res['default_port'] = highest_record['port']
    except Exception:
        res['exception'] = traceback.format_exc()
    return res