from __future__ import absolute_import, division, print_function
from itertools import product
from ansible.module_utils.basic import AnsibleModule
class ZfsDelegateAdmin(object):

    def __init__(self, module):
        self.module = module
        self.name = module.params.get('name')
        self.state = module.params.get('state')
        self.users = module.params.get('users')
        self.groups = module.params.get('groups')
        self.everyone = module.params.get('everyone')
        self.perms = module.params.get('permissions')
        self.scope = None
        self.changed = False
        self.initial_perms = None
        self.subcommand = 'allow'
        self.recursive_opt = []
        self.run_method = self.update
        self.setup(module)

    def setup(self, module):
        """ Validate params and set up for run.
        """
        if self.state == 'absent':
            self.subcommand = 'unallow'
            if module.params.get('recursive'):
                self.recursive_opt = ['-r']
        local = module.params.get('local')
        descendents = module.params.get('descendents')
        if local and descendents or (not local and (not descendents)):
            self.scope = 'ld'
        elif local:
            self.scope = 'l'
        elif descendents:
            self.scope = 'd'
        else:
            self.module.fail_json(msg='Impossible value for local and descendents')
        if not (self.users or self.groups or self.everyone):
            if self.state == 'present':
                self.module.fail_json(msg='One of `users`, `groups`, or `everyone` must be set')
            elif self.state == 'absent':
                self.run_method = self.clear
        self.zfs_path = module.get_bin_path('zfs', True)

    @property
    def current_perms(self):
        """ Parse the output of `zfs allow <name>` to retrieve current permissions.
        """
        out = self.run_zfs_raw(subcommand='allow')
        perms = {'l': {'u': {}, 'g': {}, 'e': []}, 'd': {'u': {}, 'g': {}, 'e': []}, 'ld': {'u': {}, 'g': {}, 'e': []}}
        linemap = {'Local permissions:': 'l', 'Descendent permissions:': 'd', 'Local+Descendent permissions:': 'ld'}
        scope = None
        for line in out.splitlines():
            scope = linemap.get(line, scope)
            if not scope:
                continue
            if ' (unknown: ' in line:
                line = line.replace('(unknown: ', '', 1)
                line = line.replace(')', '', 1)
            try:
                if line.startswith('\tuser ') or line.startswith('\tgroup '):
                    ent_type, ent, cur_perms = line.split()
                    perms[scope][ent_type[0]][ent] = cur_perms.split(',')
                elif line.startswith('\teveryone '):
                    perms[scope]['e'] = line.split()[1].split(',')
            except ValueError:
                self.module.fail_json(msg="Cannot parse user/group permission output by `zfs allow`: '%s'" % line)
        return perms

    def run_zfs_raw(self, subcommand=None, args=None):
        """ Run a raw zfs command, fail on error.
        """
        cmd = [self.zfs_path, subcommand or self.subcommand] + (args or []) + [self.name]
        rc, out, err = self.module.run_command(cmd)
        if rc:
            self.module.fail_json(msg='Command `%s` failed: %s' % (' '.join(cmd), err))
        return out

    def run_zfs(self, args):
        """ Run zfs allow/unallow with appropriate options as per module arguments.
        """
        args = self.recursive_opt + ['-' + self.scope] + args
        if self.perms:
            args.append(','.join(self.perms))
        return self.run_zfs_raw(args=args)

    def clear(self):
        """ Called by run() to clear all permissions.
        """
        changed = False
        stdout = ''
        for scope, ent_type in product(('ld', 'l', 'd'), ('u', 'g')):
            for ent in self.initial_perms[scope][ent_type].keys():
                stdout += self.run_zfs(['-%s' % ent_type, ent])
                changed = True
        for scope in ('ld', 'l', 'd'):
            if self.initial_perms[scope]['e']:
                stdout += self.run_zfs(['-e'])
                changed = True
        return (changed, stdout)

    def update(self):
        """ Update permissions as per module arguments.
        """
        stdout = ''
        for ent_type, entities in (('u', self.users), ('g', self.groups)):
            if entities:
                stdout += self.run_zfs(['-%s' % ent_type, ','.join(entities)])
        if self.everyone:
            stdout += self.run_zfs(['-e'])
        return (self.initial_perms != self.current_perms, stdout)

    def run(self):
        """ Run an operation, return results for Ansible.
        """
        exit_args = {'state': self.state}
        self.initial_perms = self.current_perms
        exit_args['changed'], stdout = self.run_method()
        if exit_args['changed']:
            exit_args['msg'] = 'ZFS delegated admin permissions updated'
            exit_args['stdout'] = stdout
        self.module.exit_json(**exit_args)