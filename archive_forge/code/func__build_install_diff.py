from __future__ import absolute_import, division, print_function
import re
import shlex
from ansible.module_utils.basic import AnsibleModule
from collections import defaultdict, namedtuple
def _build_install_diff(pacman_verb, pkglist):
    cmd = cmd_base + [pacman_verb, '--print-format', '%n %v'] + [p.source for p in pkglist]
    rc, stdout, stderr = self.m.run_command(cmd, check_rc=False)
    if rc != 0:
        self.fail('Failed to list package(s) to install', cmd=cmd, stdout=stdout, stderr=stderr)
    name_ver = [l.strip() for l in stdout.splitlines()]
    before = []
    after = []
    to_be_installed = []
    for p in name_ver:
        if 'loading packages' in p or 'there is nothing to do' in p or 'Avoid running' in p:
            continue
        name, version = p.split()
        if name in self.inventory['installed_pkgs']:
            before.append('%s-%s-%s' % (name, self.inventory['installed_pkgs'][name], self.inventory['pkg_reasons'][name]))
        if name in pkgs_to_set_reason:
            after.append('%s-%s-%s' % (name, version, self.m.params['reason']))
        elif name in self.inventory['pkg_reasons']:
            after.append('%s-%s-%s' % (name, version, self.inventory['pkg_reasons'][name]))
        else:
            after.append('%s-%s' % (name, version))
        to_be_installed.append(name)
    return (to_be_installed, before, after)