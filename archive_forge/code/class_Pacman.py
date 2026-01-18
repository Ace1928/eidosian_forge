from __future__ import absolute_import, division, print_function
import re
import shlex
from ansible.module_utils.basic import AnsibleModule
from collections import defaultdict, namedtuple
class Pacman(object):

    def __init__(self, module):
        self.m = module
        self.m.run_command_environ_update = dict(LC_ALL='C')
        p = self.m.params
        self._msgs = []
        self._stdouts = []
        self._stderrs = []
        self.changed = False
        self.exit_params = {}
        self.pacman_path = self.m.get_bin_path(p['executable'], True)
        self._cached_database = None
        if p['state'] == 'installed':
            self.target_state = 'present'
        elif p['state'] == 'removed':
            self.target_state = 'absent'
        else:
            self.target_state = p['state']

    def add_exit_infos(self, msg=None, stdout=None, stderr=None):
        if msg:
            self._msgs.append(msg)
        if stdout:
            self._stdouts.append(stdout)
        if stderr:
            self._stderrs.append(stderr)

    def _set_mandatory_exit_params(self):
        msg = '\n'.join(self._msgs)
        stdouts = '\n'.join(self._stdouts)
        stderrs = '\n'.join(self._stderrs)
        if stdouts:
            self.exit_params['stdout'] = stdouts
        if stderrs:
            self.exit_params['stderr'] = stderrs
        self.exit_params['msg'] = msg

    def fail(self, msg=None, stdout=None, stderr=None, **kwargs):
        self.add_exit_infos(msg, stdout, stderr)
        self._set_mandatory_exit_params()
        if kwargs:
            self.exit_params.update(**kwargs)
        self.m.fail_json(**self.exit_params)

    def success(self):
        self._set_mandatory_exit_params()
        self.m.exit_json(changed=self.changed, **self.exit_params)

    def run(self):
        if self.m.params['update_cache']:
            self.update_package_db()
            if not (self.m.params['name'] or self.m.params['upgrade']):
                self.success()
        self.inventory = self._build_inventory()
        if self.m.params['upgrade']:
            self.upgrade()
            self.success()
        if self.m.params['name']:
            pkgs = self.package_list()
            if self.target_state == 'absent':
                self.remove_packages(pkgs)
                self.success()
            else:
                self.install_packages(pkgs)
                self.success()
        self.fail('This is a bug')

    def install_packages(self, pkgs):
        pkgs_to_install = []
        pkgs_to_install_from_url = []
        pkgs_to_set_reason = []
        for p in pkgs:
            if self.m.params['reason'] and (p.name not in self.inventory['pkg_reasons'] or (self.m.params['reason_for'] == 'all' and self.inventory['pkg_reasons'][p.name] != self.m.params['reason'])):
                pkgs_to_set_reason.append(p.name)
            if p.source_is_URL:
                pkgs_to_install_from_url.append(p)
                continue
            if p.name not in self.inventory['installed_pkgs'] or (self.target_state == 'latest' and p.name in self.inventory['upgradable_pkgs']):
                pkgs_to_install.append(p)
        if len(pkgs_to_install) == 0 and len(pkgs_to_install_from_url) == 0 and (len(pkgs_to_set_reason) == 0):
            self.exit_params['packages'] = []
            self.add_exit_infos('package(s) already installed')
            return
        cmd_base = [self.pacman_path, '--noconfirm', '--noprogressbar', '--needed']
        if self.m.params['extra_args']:
            cmd_base.extend(self.m.params['extra_args'])

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
        before = []
        after = []
        installed_pkgs = []
        if pkgs_to_install:
            p, b, a = _build_install_diff('--sync', pkgs_to_install)
            installed_pkgs.extend(p)
            before.extend(b)
            after.extend(a)
        if pkgs_to_install_from_url:
            p, b, a = _build_install_diff('--upgrade', pkgs_to_install_from_url)
            installed_pkgs.extend(p)
            before.extend(b)
            after.extend(a)
        if len(installed_pkgs) == 0 and len(pkgs_to_set_reason) == 0:
            self.exit_params['packages'] = []
            self.add_exit_infos('package(s) already installed')
            return
        self.changed = True
        self.exit_params['diff'] = {'before': '\n'.join(sorted(before)) + '\n' if before else '', 'after': '\n'.join(sorted(after)) + '\n' if after else ''}
        changed_reason_pkgs = [p for p in pkgs_to_set_reason if p not in installed_pkgs]
        if self.m.check_mode:
            self.add_exit_infos('Would have installed %d packages' % (len(installed_pkgs) + len(changed_reason_pkgs)))
            self.exit_params['packages'] = sorted(installed_pkgs + changed_reason_pkgs)
            return

        def _install_packages_for_real(pacman_verb, pkglist):
            cmd = cmd_base + [pacman_verb] + [p.source for p in pkglist]
            rc, stdout, stderr = self.m.run_command(cmd, check_rc=False)
            if rc != 0:
                self.fail('Failed to install package(s)', cmd=cmd, stdout=stdout, stderr=stderr)
            self.add_exit_infos(stdout=stdout, stderr=stderr)
            self._invalidate_database()
        if pkgs_to_install:
            _install_packages_for_real('--sync', pkgs_to_install)
        if pkgs_to_install_from_url:
            _install_packages_for_real('--upgrade', pkgs_to_install_from_url)
        if pkgs_to_set_reason:
            cmd = [self.pacman_path, '--noconfirm', '--database']
            if self.m.params['reason'] == 'dependency':
                cmd.append('--asdeps')
            else:
                cmd.append('--asexplicit')
            cmd.extend(pkgs_to_set_reason)
            rc, stdout, stderr = self.m.run_command(cmd, check_rc=False)
            if rc != 0:
                self.fail('Failed to install package(s)', cmd=cmd, stdout=stdout, stderr=stderr)
            self.add_exit_infos(stdout=stdout, stderr=stderr)
        self.exit_params['packages'] = sorted(installed_pkgs + changed_reason_pkgs)
        self.add_exit_infos('Installed %d package(s)' % (len(installed_pkgs) + len(changed_reason_pkgs)))

    def remove_packages(self, pkgs):
        pkg_names_to_remove = [p.name for p in pkgs if p.name in self.inventory['installed_pkgs']]
        if len(pkg_names_to_remove) == 0:
            self.exit_params['packages'] = []
            self.add_exit_infos('package(s) already absent')
            return
        self.changed = True
        cmd_base = [self.pacman_path, '--remove', '--noconfirm', '--noprogressbar']
        cmd_base += self.m.params['extra_args']
        cmd_base += ['--nodeps', '--nodeps'] if self.m.params['force'] else []
        cmd = cmd_base + ['--print-format', '%n-%v'] + pkg_names_to_remove
        rc, stdout, stderr = self.m.run_command(cmd, check_rc=False)
        if rc != 0:
            self.fail('failed to list package(s) to remove', cmd=cmd, stdout=stdout, stderr=stderr)
        removed_pkgs = stdout.split()
        self.exit_params['packages'] = removed_pkgs
        self.exit_params['diff'] = {'before': '\n'.join(removed_pkgs) + '\n', 'after': ''}
        if self.m.check_mode:
            self.exit_params['packages'] = removed_pkgs
            self.add_exit_infos('Would have removed %d packages' % len(removed_pkgs))
            return
        nosave_args = ['--nosave'] if self.m.params['remove_nosave'] else []
        cmd = cmd_base + nosave_args + pkg_names_to_remove
        rc, stdout, stderr = self.m.run_command(cmd, check_rc=False)
        if rc != 0:
            self.fail('failed to remove package(s)', cmd=cmd, stdout=stdout, stderr=stderr)
        self._invalidate_database()
        self.exit_params['packages'] = removed_pkgs
        self.add_exit_infos('Removed %d package(s)' % len(removed_pkgs), stdout=stdout, stderr=stderr)

    def upgrade(self):
        """Runs pacman --sync --sysupgrade if there are upgradable packages"""
        if len(self.inventory['upgradable_pkgs']) == 0:
            self.add_exit_infos('Nothing to upgrade')
            return
        self.changed = True
        diff = {'before': '', 'after': ''}
        for pkg, versions in self.inventory['upgradable_pkgs'].items():
            diff['before'] += '%s-%s\n' % (pkg, versions.current)
            diff['after'] += '%s-%s\n' % (pkg, versions.latest)
        self.exit_params['diff'] = diff
        self.exit_params['packages'] = self.inventory['upgradable_pkgs'].keys()
        if self.m.check_mode:
            self.add_exit_infos('%d packages would have been upgraded' % len(self.inventory['upgradable_pkgs']))
        else:
            cmd = [self.pacman_path, '--sync', '--sysupgrade', '--quiet', '--noconfirm']
            if self.m.params['upgrade_extra_args']:
                cmd += self.m.params['upgrade_extra_args']
            rc, stdout, stderr = self.m.run_command(cmd, check_rc=False)
            self._invalidate_database()
            if rc == 0:
                self.add_exit_infos('System upgraded', stdout=stdout, stderr=stderr)
            else:
                self.fail('Could not upgrade', cmd=cmd, stdout=stdout, stderr=stderr)

    def _list_database(self):
        """runs pacman --sync --list with some caching"""
        if self._cached_database is None:
            dummy, packages, dummy = self.m.run_command([self.pacman_path, '--sync', '--list'], check_rc=True)
            self._cached_database = packages.splitlines()
        return self._cached_database

    def _invalidate_database(self):
        """invalidates the pacman --sync --list cache"""
        self._cached_database = None

    def update_package_db(self):
        """runs pacman --sync --refresh"""
        if self.m.check_mode:
            self.add_exit_infos('Would have updated the package db')
            self.changed = True
            self.exit_params['cache_updated'] = True
            return
        cmd = [self.pacman_path, '--sync', '--refresh']
        if self.m.params['update_cache_extra_args']:
            cmd += self.m.params['update_cache_extra_args']
        if self.m.params['force']:
            cmd += ['--refresh']
        else:
            pre_state = sorted(self._list_database())
        rc, stdout, stderr = self.m.run_command(cmd, check_rc=False)
        self._invalidate_database()
        if self.m.params['force']:
            self.exit_params['cache_updated'] = True
        else:
            post_state = sorted(self._list_database())
            self.exit_params['cache_updated'] = pre_state != post_state
        if self.exit_params['cache_updated']:
            self.changed = True
        if rc == 0:
            self.add_exit_infos('Updated package db', stdout=stdout, stderr=stderr)
        else:
            self.fail('could not update package db', cmd=cmd, stdout=stdout, stderr=stderr)

    def package_list(self):
        """Takes the input package list and resolves packages groups to their package list using the inventory,
        extracts package names from packages given as files or URLs using calls to pacman

        Returns the expanded/resolved list as a list of Package
        """
        pkg_list = []
        for pkg in self.m.params['name']:
            if not pkg:
                continue
            is_URL = False
            if pkg in self.inventory['available_groups']:
                for group_member in self.inventory['available_groups'][pkg]:
                    pkg_list.append(Package(name=group_member, source=group_member))
            elif pkg in self.inventory['available_pkgs'] or pkg in self.inventory['installed_pkgs']:
                pkg_list.append(Package(name=pkg, source=pkg))
            else:
                cmd = [self.pacman_path, '--sync', '--print-format', '%n', pkg]
                rc, stdout, stderr = self.m.run_command(cmd, check_rc=False)
                if rc != 0:
                    cmd = [self.pacman_path, '--upgrade', '--print-format', '%n', pkg]
                    rc, stdout, stderr = self.m.run_command(cmd, check_rc=False)
                    if rc != 0:
                        if self.target_state == 'absent':
                            continue
                        else:
                            self.fail(msg='Failed to list package %s' % pkg, cmd=cmd, stdout=stdout, stderr=stderr, rc=rc)
                    stdout = stdout.splitlines()[-1]
                    is_URL = True
                pkg_name = stdout.strip()
                pkg_list.append(Package(name=pkg_name, source=pkg, source_is_URL=is_URL))
        return pkg_list

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