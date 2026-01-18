from __future__ import absolute_import, division, print_function
from collections import defaultdict
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
class ZFSFacts(object):

    def __init__(self, module):
        self.module = module
        self.name = module.params['name']
        self.recurse = module.params['recurse']
        self.parsable = module.params['parsable']
        self.properties = module.params['properties']
        self.type = module.params['type']
        self.depth = module.params['depth']
        self._datasets = defaultdict(dict)
        self.facts = []

    def dataset_exists(self):
        cmd = [self.module.get_bin_path('zfs'), 'list', self.name]
        rc, out, err = self.module.run_command(cmd)
        if rc == 0:
            return True
        else:
            return False

    def get_facts(self):
        cmd = [self.module.get_bin_path('zfs'), 'get', '-H']
        if self.parsable:
            cmd.append('-p')
        if self.recurse:
            cmd.append('-r')
        if int(self.depth) != 0:
            cmd.append('-d')
            cmd.append('%s' % self.depth)
        if self.type:
            cmd.append('-t')
            cmd.append(self.type)
        cmd.extend(['-o', 'name,property,value', self.properties, self.name])
        rc, out, err = self.module.run_command(cmd)
        if rc == 0:
            for line in out.splitlines():
                dataset, property, value = line.split('\t')
                self._datasets[dataset].update({property: value})
            for k, v in iteritems(self._datasets):
                v.update({'name': k})
                self.facts.append(v)
            return {'ansible_zfs_datasets': self.facts}
        else:
            self.module.fail_json(msg='Error while trying to get facts about ZFS dataset: %s' % self.name, stderr=err, rc=rc)