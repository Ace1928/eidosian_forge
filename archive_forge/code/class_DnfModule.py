from __future__ import absolute_import, division, print_function
import os
import re
import sys
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.urls import fetch_file
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.yumdnf import YumDnf, yumdnf_argument_spec
class DnfModule(YumDnf):
    """
    DNF Ansible module back-end implementation
    """

    def __init__(self, module):
        super(DnfModule, self).__init__(module)
        self._ensure_dnf()
        self.lockfile = '/var/cache/dnf/*_lock.pid'
        self.pkg_mgr_name = 'dnf'
        try:
            self.with_modules = dnf.base.WITH_MODULES
        except AttributeError:
            self.with_modules = False
        self.allowerasing = self.module.params['allowerasing']
        self.nobest = self.module.params['nobest']

    def is_lockfile_pid_valid(self):
        return True

    def _sanitize_dnf_error_msg_install(self, spec, error):
        """
        For unhandled dnf.exceptions.Error scenarios, there are certain error
        messages we want to filter in an install scenario. Do that here.
        """
        if to_text('no package matched') in to_text(error) or to_text('No match for argument:') in to_text(error):
            return 'No package {0} available.'.format(spec)
        return error

    def _sanitize_dnf_error_msg_remove(self, spec, error):
        """
        For unhandled dnf.exceptions.Error scenarios, there are certain error
        messages we want to ignore in a removal scenario as known benign
        failures. Do that here.
        """
        if 'no package matched' in to_native(error) or 'No match for argument:' in to_native(error):
            return (False, '{0} is not installed'.format(spec))
        return (True, error)

    def _package_dict(self, package):
        """Return a dictionary of information for the package."""
        result = {'name': package.name, 'arch': package.arch, 'epoch': str(package.epoch), 'release': package.release, 'version': package.version, 'repo': package.repoid}
        result['envra'] = '{epoch}:{name}-{version}-{release}.{arch}'.format(**result)
        result['nevra'] = result['envra']
        if package.installtime == 0:
            result['yumstate'] = 'available'
        else:
            result['yumstate'] = 'installed'
        return result

    def _split_package_arch(self, packagename):
        redhat_rpm_arches = ['aarch64', 'alphaev56', 'alphaev5', 'alphaev67', 'alphaev6', 'alpha', 'alphapca56', 'amd64', 'armv3l', 'armv4b', 'armv4l', 'armv5tejl', 'armv5tel', 'armv5tl', 'armv6hl', 'armv6l', 'armv7hl', 'armv7hnl', 'armv7l', 'athlon', 'geode', 'i386', 'i486', 'i586', 'i686', 'ia32e', 'ia64', 'm68k', 'mips64el', 'mips64', 'mips64r6el', 'mips64r6', 'mipsel', 'mips', 'mipsr6el', 'mipsr6', 'noarch', 'pentium3', 'pentium4', 'ppc32dy4', 'ppc64iseries', 'ppc64le', 'ppc64', 'ppc64p7', 'ppc64pseries', 'ppc8260', 'ppc8560', 'ppciseries', 'ppc', 'ppcpseries', 'riscv64', 's390', 's390x', 'sh3', 'sh4a', 'sh4', 'sh', 'sparc64', 'sparc64v', 'sparc', 'sparcv8', 'sparcv9', 'sparcv9v', 'x86_64']
        name, delimiter, arch = packagename.rpartition('.')
        if name and arch and (arch in redhat_rpm_arches):
            return (name, arch)
        return (packagename, None)

    def _packagename_dict(self, packagename):
        """
        Return a dictionary of information for a package name string or None
        if the package name doesn't contain at least all NVR elements
        """
        if packagename[-4:] == '.rpm':
            packagename = packagename[:-4]
        rpm_nevr_re = re.compile('(\\S+)-(?:(\\d*):)?(.*)-(~?\\w+[\\w.+]*)')
        try:
            arch = None
            nevr, arch = self._split_package_arch(packagename)
            if arch:
                packagename = nevr
            rpm_nevr_match = rpm_nevr_re.match(packagename)
            if rpm_nevr_match:
                name, epoch, version, release = rpm_nevr_re.match(packagename).groups()
                if not version or not version.split('.')[0].isdigit():
                    return None
            else:
                return None
        except AttributeError as e:
            self.module.fail_json(msg='Error attempting to parse package: %s, %s' % (packagename, to_native(e)), rc=1, results=[])
        if not epoch:
            epoch = '0'
        if ':' in name:
            epoch_name = name.split(':')
            epoch = epoch_name[0]
            name = ''.join(epoch_name[1:])
        result = {'name': name, 'epoch': epoch, 'release': release, 'version': version}
        return result

    def _compare_evr(self, e1, v1, r1, e2, v2, r2):
        if e1 is None:
            e1 = '0'
        else:
            e1 = str(e1)
        v1 = str(v1)
        r1 = str(r1)
        if e2 is None:
            e2 = '0'
        else:
            e2 = str(e2)
        v2 = str(v2)
        r2 = str(r2)
        rc = dnf.rpm.rpm.labelCompare((e1, v1, r1), (e2, v2, r2))
        return rc

    def _ensure_dnf(self):
        locale = get_best_parsable_locale(self.module)
        os.environ['LC_ALL'] = os.environ['LC_MESSAGES'] = locale
        os.environ['LANGUAGE'] = os.environ['LANG'] = locale
        global dnf
        try:
            import dnf
            import dnf.cli
            import dnf.const
            import dnf.exceptions
            import dnf.package
            import dnf.subject
            import dnf.util
            HAS_DNF = True
        except ImportError:
            HAS_DNF = False
        if HAS_DNF:
            return
        system_interpreters = ['/usr/libexec/platform-python', '/usr/bin/python3', '/usr/bin/python2', '/usr/bin/python']
        if not has_respawned():
            interpreter = probe_interpreters_for_module(system_interpreters, 'dnf')
            if interpreter:
                respawn_module(interpreter)
        self.module.fail_json(msg='Could not import the dnf python module using {0} ({1}). Please install `python3-dnf` or `python2-dnf` package or ensure you have specified the correct ansible_python_interpreter. (attempted {2})'.format(sys.executable, sys.version.replace('\n', ''), system_interpreters), results=[])

    def _configure_base(self, base, conf_file, disable_gpg_check, installroot='/', sslverify=True):
        """Configure the dnf Base object."""
        conf = base.conf
        if conf_file:
            if not os.access(conf_file, os.R_OK):
                self.module.fail_json(msg='cannot read configuration file', conf_file=conf_file, results=[])
            else:
                conf.config_file_path = conf_file
        conf.read()
        conf.debuglevel = 0
        conf.gpgcheck = not disable_gpg_check
        conf.localpkg_gpgcheck = not disable_gpg_check
        conf.assumeyes = True
        conf.sslverify = sslverify
        conf.installroot = installroot
        conf.substitutions.update_from_etc(installroot)
        if self.exclude:
            _excludes = list(conf.exclude)
            _excludes.extend(self.exclude)
            conf.exclude = _excludes
        if self.disable_excludes:
            _disable_excludes = list(conf.disable_excludes)
            if self.disable_excludes not in _disable_excludes:
                _disable_excludes.append(self.disable_excludes)
                conf.disable_excludes = _disable_excludes
        if self.releasever is not None:
            conf.substitutions['releasever'] = self.releasever
        if conf.substitutions.get('releasever') is None:
            self.module.warn('Unable to detect release version (use "releasever" option to specify release version)')
            conf.substitutions['releasever'] = ''
        if self.skip_broken:
            conf.strict = 0
        if self.nobest:
            conf.best = 0
        if self.download_only:
            conf.downloadonly = True
            if self.download_dir:
                conf.destdir = self.download_dir
        if self.cacheonly:
            conf.cacheonly = True
        conf.clean_requirements_on_remove = self.autoremove
        conf.install_weak_deps = self.install_weak_deps

    def _specify_repositories(self, base, disablerepo, enablerepo):
        """Enable and disable repositories matching the provided patterns."""
        base.read_all_repos()
        repos = base.repos
        for repo_pattern in disablerepo:
            if repo_pattern:
                for repo in repos.get_matching(repo_pattern):
                    repo.disable()
        for repo_pattern in enablerepo:
            if repo_pattern:
                for repo in repos.get_matching(repo_pattern):
                    repo.enable()

    def _base(self, conf_file, disable_gpg_check, disablerepo, enablerepo, installroot, sslverify):
        """Return a fully configured dnf Base object."""
        base = dnf.Base()
        self._configure_base(base, conf_file, disable_gpg_check, installroot, sslverify)
        try:
            base.setup_loggers()
        except AttributeError:
            pass
        try:
            base.init_plugins(set(self.disable_plugin), set(self.enable_plugin))
            base.pre_configure_plugins()
        except AttributeError:
            pass
        self._specify_repositories(base, disablerepo, enablerepo)
        try:
            base.configure_plugins()
        except AttributeError:
            pass
        try:
            if self.update_cache:
                try:
                    base.update_cache()
                except dnf.exceptions.RepoError as e:
                    self.module.fail_json(msg='{0}'.format(to_text(e)), results=[], rc=1)
            base.fill_sack(load_system_repo='auto')
        except dnf.exceptions.RepoError as e:
            self.module.fail_json(msg='{0}'.format(to_text(e)), results=[], rc=1)
        add_security_filters = getattr(base, 'add_security_filters', None)
        if callable(add_security_filters):
            filters = {}
            if self.bugfix:
                filters.setdefault('types', []).append('bugfix')
            if self.security:
                filters.setdefault('types', []).append('security')
            if filters:
                add_security_filters('eq', **filters)
        else:
            filters = []
            if self.bugfix:
                key = {'advisory_type__eq': 'bugfix'}
                filters.append(base.sack.query().upgrades().filter(**key))
            if self.security:
                key = {'advisory_type__eq': 'security'}
                filters.append(base.sack.query().upgrades().filter(**key))
            if filters:
                base._update_security_filters = filters
        return base

    def list_items(self, command):
        """List package info based on the command."""
        if command == 'updates':
            command = 'upgrades'
        if command in ['installed', 'upgrades', 'available']:
            results = [self._package_dict(package) for package in getattr(self.base.sack.query(), command)()]
        elif command in ['repos', 'repositories']:
            results = [{'repoid': repo.id, 'state': 'enabled'} for repo in self.base.repos.iter_enabled()]
        else:
            packages = dnf.subject.Subject(command).get_best_query(self.base.sack)
            results = [self._package_dict(package) for package in packages]
        self.module.exit_json(msg='', results=results)

    def _is_installed(self, pkg):
        installed = self.base.sack.query().installed()
        package_spec = {}
        name, arch = self._split_package_arch(pkg)
        if arch:
            package_spec['arch'] = arch
        package_details = self._packagename_dict(pkg)
        if package_details:
            package_details['epoch'] = int(package_details['epoch'])
            package_spec.update(package_details)
        else:
            package_spec['name'] = name
        return bool(installed.filter(**package_spec))

    def _is_newer_version_installed(self, pkg_name):
        candidate_pkg = self._packagename_dict(pkg_name)
        if not candidate_pkg:
            return False
        installed = self.base.sack.query().installed()
        installed_pkg = installed.filter(name=candidate_pkg['name']).run()
        if installed_pkg:
            installed_pkg = installed_pkg[0]
            evr_cmp = self._compare_evr(installed_pkg.epoch, installed_pkg.version, installed_pkg.release, candidate_pkg['epoch'], candidate_pkg['version'], candidate_pkg['release'])
            return evr_cmp == 1
        else:
            return False

    def _mark_package_install(self, pkg_spec, upgrade=False):
        """Mark the package for install."""
        is_newer_version_installed = self._is_newer_version_installed(pkg_spec)
        is_installed = self._is_installed(pkg_spec)
        try:
            if is_newer_version_installed:
                if self.allow_downgrade:
                    if upgrade:
                        if is_installed:
                            self.base.upgrade(pkg_spec)
                        else:
                            self.base.install(pkg_spec, strict=self.base.conf.strict)
                    else:
                        self.base.install(pkg_spec, strict=self.base.conf.strict)
                else:
                    pass
            elif is_installed:
                if upgrade:
                    self.base.upgrade(pkg_spec)
                else:
                    pass
            else:
                self.base.install(pkg_spec, strict=self.base.conf.strict)
            return {'failed': False, 'msg': '', 'failure': '', 'rc': 0}
        except dnf.exceptions.MarkingError as e:
            return {'failed': True, 'msg': 'No package {0} available.'.format(pkg_spec), 'failure': ' '.join((pkg_spec, to_native(e))), 'rc': 1, 'results': []}
        except dnf.exceptions.DepsolveError as e:
            return {'failed': True, 'msg': 'Depsolve Error occurred for package {0}.'.format(pkg_spec), 'failure': ' '.join((pkg_spec, to_native(e))), 'rc': 1, 'results': []}
        except dnf.exceptions.Error as e:
            if to_text('already installed') in to_text(e):
                return {'failed': False, 'msg': '', 'failure': ''}
            else:
                return {'failed': True, 'msg': 'Unknown Error occurred for package {0}.'.format(pkg_spec), 'failure': ' '.join((pkg_spec, to_native(e))), 'rc': 1, 'results': []}

    def _whatprovides(self, filepath):
        self.base.read_all_repos()
        available = self.base.sack.query().available()
        files_filter = available.filter(file=filepath)
        pkg_spec = files_filter.union(available.filter(provides=filepath)).run()
        if pkg_spec:
            return pkg_spec[0].name

    def _parse_spec_group_file(self):
        pkg_specs, grp_specs, module_specs, filenames = ([], [], [], [])
        already_loaded_comps = False
        for name in self.names:
            if '://' in name:
                name = fetch_file(self.module, name)
                filenames.append(name)
            elif name.endswith('.rpm'):
                filenames.append(name)
            elif name.startswith('/'):
                pkg_spec = self._whatprovides(name)
                if pkg_spec:
                    pkg_specs.append(pkg_spec)
                    continue
            elif name.startswith('@') or '/' in name:
                if not already_loaded_comps:
                    self.base.read_comps()
                    already_loaded_comps = True
                grp_env_mdl_candidate = name[1:].strip()
                if self.with_modules:
                    mdl = self.module_base._get_modules(grp_env_mdl_candidate)
                    if mdl[0]:
                        module_specs.append(grp_env_mdl_candidate)
                    else:
                        grp_specs.append(grp_env_mdl_candidate)
                else:
                    grp_specs.append(grp_env_mdl_candidate)
            else:
                pkg_specs.append(name)
        return (pkg_specs, grp_specs, module_specs, filenames)

    def _update_only(self, pkgs):
        not_installed = []
        for pkg in pkgs:
            if self._is_installed(self._package_dict(pkg)['nevra'] if isinstance(pkg, dnf.package.Package) else pkg):
                try:
                    if isinstance(pkg, dnf.package.Package):
                        self.base.package_upgrade(pkg)
                    else:
                        self.base.upgrade(pkg)
                except Exception as e:
                    self.module.fail_json(msg='Error occurred attempting update_only operation: {0}'.format(to_native(e)), results=[], rc=1)
            else:
                not_installed.append(pkg)
        return not_installed

    def _install_remote_rpms(self, filenames):
        if int(dnf.__version__.split('.')[0]) >= 2:
            pkgs = list(sorted(self.base.add_remote_rpms(list(filenames)), reverse=True))
        else:
            pkgs = []
            try:
                for filename in filenames:
                    pkgs.append(self.base.add_remote_rpm(filename))
            except IOError as e:
                if to_text('Can not load RPM file') in to_text(e):
                    self.module.fail_json(msg='Error occurred attempting remote rpm install of package: {0}. {1}'.format(filename, to_native(e)), results=[], rc=1)
        if self.update_only:
            self._update_only(pkgs)
        else:
            for pkg in pkgs:
                try:
                    if self._is_newer_version_installed(self._package_dict(pkg)['nevra']):
                        if self.allow_downgrade:
                            self.base.package_install(pkg, strict=self.base.conf.strict)
                    else:
                        self.base.package_install(pkg, strict=self.base.conf.strict)
                except Exception as e:
                    self.module.fail_json(msg='Error occurred attempting remote rpm operation: {0}'.format(to_native(e)), results=[], rc=1)

    def _is_module_installed(self, module_spec):
        if self.with_modules:
            module_spec = module_spec.strip()
            module_list, nsv = self.module_base._get_modules(module_spec)
            enabled_streams = self.base._moduleContainer.getEnabledStream(nsv.name)
            if enabled_streams:
                if nsv.stream:
                    if nsv.stream in enabled_streams:
                        return True
                    else:
                        return False
                else:
                    return True
        return False

    def ensure(self):
        response = {'msg': '', 'changed': False, 'results': [], 'rc': 0}
        failure_response = {'msg': '', 'failures': [], 'results': [], 'rc': 1}
        if not self.names and self.autoremove:
            self.names = []
            self.state = 'absent'
        if self.names == ['*'] and self.state == 'latest':
            try:
                self.base.upgrade_all()
            except dnf.exceptions.DepsolveError as e:
                failure_response['msg'] = 'Depsolve Error occurred attempting to upgrade all packages'
                self.module.fail_json(**failure_response)
        else:
            pkg_specs, group_specs, module_specs, filenames = self._parse_spec_group_file()
            pkg_specs = [p.strip() for p in pkg_specs]
            filenames = [f.strip() for f in filenames]
            groups = []
            environments = []
            for group_spec in (g.strip() for g in group_specs):
                group = self.base.comps.group_by_pattern(group_spec)
                if group:
                    groups.append(group.id)
                else:
                    environment = self.base.comps.environment_by_pattern(group_spec)
                    if environment:
                        environments.append(environment.id)
                    else:
                        self.module.fail_json(msg='No group {0} available.'.format(group_spec), results=[])
            if self.state in ['installed', 'present']:
                self._install_remote_rpms(filenames)
                for filename in filenames:
                    response['results'].append('Installed {0}'.format(filename))
                if module_specs and self.with_modules:
                    for module in module_specs:
                        try:
                            if not self._is_module_installed(module):
                                response['results'].append('Module {0} installed.'.format(module))
                            self.module_base.install([module])
                            self.module_base.enable([module])
                        except dnf.exceptions.MarkingErrors as e:
                            failure_response['failures'].append(' '.join((module, to_native(e))))
                for group in groups:
                    try:
                        group_pkg_count_installed = self.base.group_install(group, dnf.const.GROUP_PACKAGE_TYPES)
                        if group_pkg_count_installed == 0:
                            response['results'].append('Group {0} already installed.'.format(group))
                        else:
                            response['results'].append('Group {0} installed.'.format(group))
                    except dnf.exceptions.DepsolveError as e:
                        failure_response['msg'] = 'Depsolve Error occurred attempting to install group: {0}'.format(group)
                        self.module.fail_json(**failure_response)
                    except dnf.exceptions.Error as e:
                        failure_response['failures'].append(' '.join((group, to_native(e))))
                for environment in environments:
                    try:
                        self.base.environment_install(environment, dnf.const.GROUP_PACKAGE_TYPES)
                    except dnf.exceptions.DepsolveError as e:
                        failure_response['msg'] = 'Depsolve Error occurred attempting to install environment: {0}'.format(environment)
                        self.module.fail_json(**failure_response)
                    except dnf.exceptions.Error as e:
                        failure_response['failures'].append(' '.join((environment, to_native(e))))
                if module_specs and (not self.with_modules):
                    self.module.fail_json(msg='No group {0} available.'.format(module_specs[0]), results=[])
                if self.update_only:
                    not_installed = self._update_only(pkg_specs)
                    for spec in not_installed:
                        response['results'].append('Packages providing %s not installed due to update_only specified' % spec)
                else:
                    for pkg_spec in pkg_specs:
                        install_result = self._mark_package_install(pkg_spec)
                        if install_result['failed']:
                            if install_result['msg']:
                                failure_response['msg'] += install_result['msg']
                            failure_response['failures'].append(self._sanitize_dnf_error_msg_install(pkg_spec, install_result['failure']))
                        elif install_result['msg']:
                            response['results'].append(install_result['msg'])
            elif self.state == 'latest':
                self._install_remote_rpms(filenames)
                for filename in filenames:
                    response['results'].append('Installed {0}'.format(filename))
                if module_specs and self.with_modules:
                    for module in module_specs:
                        try:
                            if self._is_module_installed(module):
                                response['results'].append('Module {0} upgraded.'.format(module))
                            self.module_base.upgrade([module])
                        except dnf.exceptions.MarkingErrors as e:
                            failure_response['failures'].append(' '.join((module, to_native(e))))
                for group in groups:
                    try:
                        try:
                            self.base.group_upgrade(group)
                            response['results'].append('Group {0} upgraded.'.format(group))
                        except dnf.exceptions.CompsError:
                            if not self.update_only:
                                group_pkg_count_installed = self.base.group_install(group, dnf.const.GROUP_PACKAGE_TYPES)
                                if group_pkg_count_installed == 0:
                                    response['results'].append('Group {0} already installed.'.format(group))
                                else:
                                    response['results'].append('Group {0} installed.'.format(group))
                    except dnf.exceptions.Error as e:
                        failure_response['failures'].append(' '.join((group, to_native(e))))
                for environment in environments:
                    try:
                        try:
                            self.base.environment_upgrade(environment)
                        except dnf.exceptions.CompsError:
                            self.base.environment_install(environment, dnf.const.GROUP_PACKAGE_TYPES)
                    except dnf.exceptions.DepsolveError as e:
                        failure_response['msg'] = 'Depsolve Error occurred attempting to install environment: {0}'.format(environment)
                    except dnf.exceptions.Error as e:
                        failure_response['failures'].append(' '.join((environment, to_native(e))))
                if self.update_only:
                    not_installed = self._update_only(pkg_specs)
                    for spec in not_installed:
                        response['results'].append('Packages providing %s not installed due to update_only specified' % spec)
                else:
                    for pkg_spec in pkg_specs:
                        self.base.conf.best = not self.nobest
                        install_result = self._mark_package_install(pkg_spec, upgrade=True)
                        if install_result['failed']:
                            if install_result['msg']:
                                failure_response['msg'] += install_result['msg']
                            failure_response['failures'].append(self._sanitize_dnf_error_msg_install(pkg_spec, install_result['failure']))
                        elif install_result['msg']:
                            response['results'].append(install_result['msg'])
            else:
                if filenames:
                    self.module.fail_json(msg='Cannot remove paths -- please specify package name.', results=[])
                if module_specs and self.with_modules:
                    for module in module_specs:
                        try:
                            if self._is_module_installed(module):
                                response['results'].append('Module {0} removed.'.format(module))
                            self.module_base.remove([module])
                            self.module_base.disable([module])
                            self.module_base.reset([module])
                        except dnf.exceptions.MarkingErrors as e:
                            failure_response['failures'].append(' '.join((module, to_native(e))))
                for group in groups:
                    try:
                        self.base.group_remove(group)
                    except dnf.exceptions.CompsError:
                        pass
                    except AttributeError:
                        pass
                for environment in environments:
                    try:
                        self.base.environment_remove(environment)
                    except dnf.exceptions.CompsError:
                        pass
                installed = self.base.sack.query().installed()
                for pkg_spec in pkg_specs:
                    if '*' in pkg_spec:
                        try:
                            self.base.remove(pkg_spec)
                        except dnf.exceptions.MarkingError as e:
                            is_failure, handled_remove_error = self._sanitize_dnf_error_msg_remove(pkg_spec, to_native(e))
                            if is_failure:
                                failure_response['failures'].append('{0} - {1}'.format(pkg_spec, to_native(e)))
                            else:
                                response['results'].append(handled_remove_error)
                        continue
                    installed_pkg = dnf.subject.Subject(pkg_spec).get_best_query(sack=self.base.sack).installed().run()
                    for pkg in installed_pkg:
                        self.base.remove(str(pkg))
                self.allowerasing = True
                if self.autoremove:
                    self.base.autoremove()
        try:
            if not self.base.resolve(allow_erasing=self.allowerasing):
                if failure_response['failures']:
                    failure_response['msg'] = 'Failed to install some of the specified packages'
                    self.module.fail_json(**failure_response)
                response['msg'] = 'Nothing to do'
                self.module.exit_json(**response)
            else:
                response['changed'] = True
                if self.download_only:
                    install_action = 'Downloaded'
                else:
                    install_action = 'Installed'
                for package in self.base.transaction.install_set:
                    response['results'].append('{0}: {1}'.format(install_action, package))
                for package in self.base.transaction.remove_set:
                    response['results'].append('Removed: {0}'.format(package))
                if failure_response['failures']:
                    failure_response['msg'] = 'Failed to install some of the specified packages'
                    self.module.fail_json(**failure_response)
                if self.module.check_mode:
                    response['msg'] = 'Check mode: No changes made, but would have if not in check mode'
                    self.module.exit_json(**response)
                try:
                    if self.download_only and self.download_dir and self.base.conf.destdir:
                        dnf.util.ensure_dir(self.base.conf.destdir)
                        self.base.repos.all().pkgdir = self.base.conf.destdir
                    self.base.download_packages(self.base.transaction.install_set)
                except dnf.exceptions.DownloadError as e:
                    self.module.fail_json(msg='Failed to download packages: {0}'.format(to_text(e)), results=[])
                if not self.disable_gpg_check:
                    for package in self.base.transaction.install_set:
                        fail = False
                        gpgres, gpgerr = self.base._sig_check_pkg(package)
                        if gpgres == 0:
                            continue
                        elif gpgres == 1:
                            try:
                                self.base._get_key_for_package(package)
                            except dnf.exceptions.Error as e:
                                fail = True
                        else:
                            fail = True
                        if fail:
                            msg = 'Failed to validate GPG signature for {0}: {1}'.format(package, gpgerr)
                            self.module.fail_json(msg)
                if self.download_only:
                    self.module.exit_json(**response)
                else:
                    tid = self.base.do_transaction()
                    if tid is not None:
                        transaction = self.base.history.old([tid])[0]
                        if transaction.return_code:
                            failure_response['failures'].append(transaction.output())
                if failure_response['failures']:
                    failure_response['msg'] = 'Failed to install some of the specified packages'
                    self.module.fail_json(**failure_response)
                self.module.exit_json(**response)
        except dnf.exceptions.DepsolveError as e:
            failure_response['msg'] = 'Depsolve Error occurred: {0}'.format(to_native(e))
            self.module.fail_json(**failure_response)
        except dnf.exceptions.Error as e:
            if to_text('already installed') in to_text(e):
                response['changed'] = False
                response['results'].append('Package already installed: {0}'.format(to_native(e)))
                self.module.exit_json(**response)
            else:
                failure_response['msg'] = 'Unknown Error occurred: {0}'.format(to_native(e))
                self.module.fail_json(**failure_response)

    def run(self):
        """The main function."""
        if self.autoremove:
            if LooseVersion(dnf.__version__) < LooseVersion('2.0.1'):
                self.module.fail_json(msg='Autoremove requires dnf>=2.0.1. Current dnf version is %s' % dnf.__version__, results=[])
        if self.download_dir:
            if LooseVersion(dnf.__version__) < LooseVersion('2.6.2'):
                self.module.fail_json(msg='download_dir requires dnf>=2.6.2. Current dnf version is %s' % dnf.__version__, results=[])
        if self.update_cache and (not self.names) and (not self.list):
            self.base = self._base(self.conf_file, self.disable_gpg_check, self.disablerepo, self.enablerepo, self.installroot, self.sslverify)
            self.module.exit_json(msg='Cache updated', changed=False, results=[], rc=0)
        if self.state is None:
            self.state = 'installed'
        if self.list:
            self.base = self._base(self.conf_file, self.disable_gpg_check, self.disablerepo, self.enablerepo, self.installroot, self.sslverify)
            self.list_items(self.list)
        else:
            if not self.download_only and (not dnf.util.am_i_root()):
                self.module.fail_json(msg='This command has to be run under the root user.', results=[])
            self.base = self._base(self.conf_file, self.disable_gpg_check, self.disablerepo, self.enablerepo, self.installroot, self.sslverify)
            if self.with_modules:
                self.module_base = dnf.module.module_base.ModuleBase(self.base)
            try:
                self.ensure()
            finally:
                self.base.close()