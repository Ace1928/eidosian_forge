from __future__ import absolute_import, division, print_function
from collections import defaultdict
from ansible.module_utils.six import iteritems
from ansible.module_utils.basic import AnsibleModule
class ZPoolFacts(object):

    def __init__(self, module):
        self.module = module
        self.name = module.params['name']
        self.parsable = module.params['parsable']
        self.properties = module.params['properties']
        self._pools = defaultdict(dict)
        self.facts = []

    def pool_exists(self):
        cmd = [self.module.get_bin_path('zpool'), 'list', self.name]
        rc, dummy, dummy = self.module.run_command(cmd)
        return rc == 0

    def get_facts(self):
        cmd = [self.module.get_bin_path('zpool'), 'get', '-H']
        if self.parsable:
            cmd.append('-p')
        cmd.append('-o')
        cmd.append('name,property,value')
        cmd.append(self.properties)
        if self.name:
            cmd.append(self.name)
        rc, out, err = self.module.run_command(cmd, check_rc=True)
        for line in out.splitlines():
            pool, prop, value = line.split('\t')
            self._pools[pool].update({prop: value})
        for k, v in iteritems(self._pools):
            v.update({'name': k})
            self.facts.append(v)
        return {'ansible_zfs_pools': self.facts}