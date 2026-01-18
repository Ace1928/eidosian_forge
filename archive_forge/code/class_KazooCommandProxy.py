from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes
class KazooCommandProxy:

    def __init__(self, module):
        self.module = module
        self.zk = KazooClient(module.params['hosts'], use_ssl=module.params['use_tls'])

    def absent(self):
        return self._absent(self.module.params['name'])

    def exists(self, znode):
        return self.zk.exists(znode)

    def list(self):
        children = self.zk.get_children(self.module.params['name'])
        return (True, {'count': len(children), 'items': children, 'msg': 'Retrieved znodes in path.', 'znode': self.module.params['name']})

    def present(self):
        return self._present(self.module.params['name'], self.module.params['value'])

    def get(self):
        return self._get(self.module.params['name'])

    def shutdown(self):
        self.zk.stop()
        self.zk.close()

    def start(self):
        self.zk.start()
        if self.module.params['auth_credential']:
            self.zk.add_auth(self.module.params['auth_scheme'], self.module.params['auth_credential'])

    def wait(self):
        return self._wait(self.module.params['name'], self.module.params['timeout'])

    def _absent(self, znode):
        if self.exists(znode):
            self.zk.delete(znode, recursive=self.module.params['recursive'])
            return (True, {'changed': True, 'msg': 'The znode was deleted.'})
        else:
            return (True, {'changed': False, 'msg': 'The znode does not exist.'})

    def _get(self, path):
        if self.exists(path):
            value, zstat = self.zk.get(path)
            stat_dict = {}
            for i in dir(zstat):
                if not i.startswith('_'):
                    attr = getattr(zstat, i)
                    if isinstance(attr, (int, str)):
                        stat_dict[i] = attr
            result = (True, {'msg': 'The node was retrieved.', 'znode': path, 'value': value, 'stat': stat_dict})
        else:
            result = (False, {'msg': 'The requested node does not exist.'})
        return result

    def _present(self, path, value):
        if self.exists(path):
            current_value, zstat = self.zk.get(path)
            if value != current_value:
                self.zk.set(path, to_bytes(value))
                return (True, {'changed': True, 'msg': 'Updated the znode value.', 'znode': path, 'value': value})
            else:
                return (True, {'changed': False, 'msg': 'No changes were necessary.', 'znode': path, 'value': value})
        else:
            self.zk.create(path, to_bytes(value), makepath=True)
            return (True, {'changed': True, 'msg': 'Created a new znode.', 'znode': path, 'value': value})

    def _wait(self, path, timeout, interval=5):
        lim = time.time() + timeout
        while time.time() < lim:
            if self.exists(path):
                return (True, {'msg': 'The node appeared before the configured timeout.', 'znode': path, 'timeout': timeout})
            else:
                time.sleep(interval)
        return (False, {'msg': 'The node did not appear before the operation timed out.', 'timeout': timeout, 'znode': path})