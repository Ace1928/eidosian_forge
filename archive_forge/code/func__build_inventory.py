from __future__ import absolute_import, division, print_function
import re
import shlex
from ansible.module_utils.basic import AnsibleModule
from collections import defaultdict, namedtuple
def _build_inventory(self):
    """Build a cache datastructure used for all pkg lookups
        Returns a dict:
        {
            "installed_pkgs": {pkgname: version},
            "installed_groups": {groupname: set(pkgnames)},
            "available_pkgs": {pkgname: version},
            "available_groups": {groupname: set(pkgnames)},
            "upgradable_pkgs": {pkgname: (current_version,latest_version)},
            "pkg_reasons": {pkgname: reason},
        }

        Fails the module if a package requested for install cannot be found
        """
    installed_pkgs = {}
    dummy, stdout, dummy = self.m.run_command([self.pacman_path, '--query'], check_rc=True)
    query_re = re.compile('^\\s*(?P<pkg>\\S+)\\s+(?P<ver>\\S+)\\s*$')
    for l in stdout.splitlines():
        query_match = query_re.match(l)
        if not query_match:
            continue
        pkg, ver = query_match.groups()
        installed_pkgs[pkg] = ver
    installed_groups = defaultdict(set)
    dummy, stdout, dummy = self.m.run_command([self.pacman_path, '--query', '--groups'], check_rc=True)
    query_groups_re = re.compile('^\\s*(?P<group>\\S+)\\s+(?P<pkg>\\S+)\\s*$')
    for l in stdout.splitlines():
        query_groups_match = query_groups_re.match(l)
        if not query_groups_match:
            continue
        group, pkgname = query_groups_match.groups()
        installed_groups[group].add(pkgname)
    available_pkgs = {}
    database = self._list_database()
    for l in database:
        l = l.strip()
        if not l:
            continue
        repo, pkg, ver = l.split()[:3]
        available_pkgs[pkg] = ver
    available_groups = defaultdict(set)
    dummy, stdout, dummy = self.m.run_command([self.pacman_path, '--sync', '--groups', '--groups'], check_rc=True)
    sync_groups_re = re.compile('^\\s*(?P<group>\\S+)\\s+(?P<pkg>\\S+)\\s*$')
    for l in stdout.splitlines():
        sync_groups_match = sync_groups_re.match(l)
        if not sync_groups_match:
            continue
        group, pkg = sync_groups_match.groups()
        available_groups[group].add(pkg)
    upgradable_pkgs = {}
    rc, stdout, stderr = self.m.run_command([self.pacman_path, '--query', '--upgrades'], check_rc=False)
    stdout = stdout.splitlines()
    if stdout and 'Avoid running' in stdout[0]:
        stdout = stdout[1:]
    stdout = '\n'.join(stdout)
    if rc == 1 and (not stdout):
        pass
    elif rc == 0:
        for l in stdout.splitlines():
            l = l.strip()
            if not l:
                continue
            if '[ignored]' in l or 'Avoid running' in l:
                continue
            s = l.split()
            if len(s) != 4:
                self.fail(msg='Invalid line: %s' % l)
            pkg = s[0]
            current = s[1]
            latest = s[3]
            upgradable_pkgs[pkg] = VersionTuple(current=current, latest=latest)
    else:
        self.fail("Couldn't get list of packages available for upgrade", stdout=stdout, stderr=stderr, rc=rc)
    pkg_reasons = {}
    dummy, stdout, dummy = self.m.run_command([self.pacman_path, '--query', '--explicit'], check_rc=True)
    for l in stdout.splitlines():
        l = l.strip()
        if not l:
            continue
        pkg = l.split()[0]
        pkg_reasons[pkg] = 'explicit'
    dummy, stdout, dummy = self.m.run_command([self.pacman_path, '--query', '--deps'], check_rc=True)
    for l in stdout.splitlines():
        l = l.strip()
        if not l:
            continue
        pkg = l.split()[0]
        pkg_reasons[pkg] = 'dependency'
    return dict(installed_pkgs=installed_pkgs, installed_groups=installed_groups, available_pkgs=available_pkgs, available_groups=available_groups, upgradable_pkgs=upgradable_pkgs, pkg_reasons=pkg_reasons)