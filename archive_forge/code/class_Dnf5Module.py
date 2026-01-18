from __future__ import absolute_import, division, print_function
import os
import sys
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.yumdnf import YumDnf, yumdnf_argument_spec
class Dnf5Module(YumDnf):

    def __init__(self, module):
        super(Dnf5Module, self).__init__(module)
        self._ensure_dnf()
        self.lockfile = ''
        self.pkg_mgr_name = 'dnf5'
        self.allowerasing = self.module.params['allowerasing']
        self.nobest = self.module.params['nobest']

    def _ensure_dnf(self):
        locale = get_best_parsable_locale(self.module)
        os.environ['LC_ALL'] = os.environ['LC_MESSAGES'] = locale
        os.environ['LANGUAGE'] = os.environ['LANG'] = locale
        global libdnf5
        has_dnf = True
        try:
            import libdnf5
        except ImportError:
            has_dnf = False
        if has_dnf:
            return
        system_interpreters = ['/usr/libexec/platform-python', '/usr/bin/python3', '/usr/bin/python2', '/usr/bin/python']
        if not has_respawned():
            interpreter = probe_interpreters_for_module(system_interpreters, 'libdnf5')
            if interpreter:
                respawn_module(interpreter)
        self.module.fail_json(msg='Could not import the libdnf5 python module using {0} ({1}). Please install python3-libdnf5 package or ensure you have specified the correct ansible_python_interpreter. (attempted {2})'.format(sys.executable, sys.version.replace('\n', ''), system_interpreters), failures=[])

    def is_lockfile_pid_valid(self):
        return True

    def run(self):
        if sys.version_info.major < 3:
            self.module.fail_json(msg='The dnf5 module requires Python 3.', failures=[], rc=1)
        if not self.list and (not self.download_only) and (os.geteuid() != 0):
            self.module.fail_json(msg='This command has to be run under the root user.', failures=[], rc=1)
        if self.enable_plugin or self.disable_plugin:
            self.module.fail_json(msg='enable_plugin and disable_plugin options are not yet implemented in DNF5', failures=[], rc=1)
        base = libdnf5.base.Base()
        conf = base.get_config()
        if self.conf_file:
            conf.config_file_path = self.conf_file
        try:
            base.load_config()
        except RuntimeError as e:
            self.module.fail_json(msg=str(e), conf_file=self.conf_file, failures=[], rc=1)
        if self.releasever is not None:
            variables = base.get_vars()
            variables.set('releasever', self.releasever)
        if self.exclude:
            conf.excludepkgs = self.exclude
        if self.disable_excludes:
            if self.disable_excludes == 'all':
                self.disable_excludes = '*'
            conf.disable_excludes = self.disable_excludes
        conf.skip_broken = self.skip_broken
        conf.best = not self.nobest
        conf.install_weak_deps = self.install_weak_deps
        conf.gpgcheck = not self.disable_gpg_check
        conf.localpkg_gpgcheck = not self.disable_gpg_check
        conf.sslverify = self.sslverify
        conf.clean_requirements_on_remove = self.autoremove
        conf.installroot = self.installroot
        conf.use_host_config = True
        conf.cacheonly = 'all' if self.cacheonly else 'none'
        if self.download_dir:
            conf.destdir = self.download_dir
        base.setup()
        log_router = base.get_logger()
        global_logger = libdnf5.logger.GlobalLogger()
        global_logger.set(log_router.get(), libdnf5.logger.Logger.Level_DEBUG)
        logger = libdnf5.logger.create_file_logger(base, 'dnf5.log')
        log_router.add_logger(logger)
        if self.update_cache:
            repo_query = libdnf5.repo.RepoQuery(base)
            repo_query.filter_type(libdnf5.repo.Repo.Type_AVAILABLE)
            for repo in repo_query:
                repo_dir = repo.get_cachedir()
                if os.path.exists(repo_dir):
                    repo_cache = libdnf5.repo.RepoCache(base, repo_dir)
                    repo_cache.write_attribute(libdnf5.repo.RepoCache.ATTRIBUTE_EXPIRED)
        sack = base.get_repo_sack()
        sack.create_repos_from_system_configuration()
        repo_query = libdnf5.repo.RepoQuery(base)
        if self.disablerepo:
            repo_query.filter_id(self.disablerepo, libdnf5.common.QueryCmp_IGLOB)
            for repo in repo_query:
                repo.disable()
        if self.enablerepo:
            repo_query.filter_id(self.enablerepo, libdnf5.common.QueryCmp_IGLOB)
            for repo in repo_query:
                repo.enable()
        try:
            sack.load_repos()
        except AttributeError:
            sack.update_and_load_enabled_repos(True)
        if self.update_cache and (not self.names) and (not self.list):
            self.module.exit_json(msg='Cache updated', changed=False, results=[], rc=0)
        if self.list:
            command = self.list
            if command == 'updates':
                command = 'upgrades'
            if command in {'installed', 'upgrades', 'available'}:
                query = libdnf5.rpm.PackageQuery(base)
                getattr(query, 'filter_{}'.format(command))()
                results = [package_to_dict(package) for package in query]
            elif command in {'repos', 'repositories'}:
                query = libdnf5.repo.RepoQuery(base)
                query.filter_enabled(True)
                results = [{'repoid': repo.get_id(), 'state': 'enabled'} for repo in query]
            else:
                resolve_spec_settings = libdnf5.base.ResolveSpecSettings()
                query = libdnf5.rpm.PackageQuery(base)
                query.resolve_pkg_spec(command, resolve_spec_settings, True)
                results = [package_to_dict(package) for package in query]
            self.module.exit_json(msg='', results=results, rc=0)
        settings = libdnf5.base.GoalJobSettings()
        try:
            settings.set_group_with_name(True)
        except AttributeError:
            settings.group_with_name = True
        if self.bugfix or self.security:
            advisory_query = libdnf5.advisory.AdvisoryQuery(base)
            types = []
            if self.bugfix:
                types.append('bugfix')
            if self.security:
                types.append('security')
            advisory_query.filter_type(types)
            settings.set_advisory_filter(advisory_query)
        goal = libdnf5.base.Goal(base)
        results = []
        if self.names == ['*'] and self.state == 'latest':
            goal.add_rpm_upgrade(settings)
        elif self.state in {'install', 'present', 'latest'}:
            upgrade = self.state == 'latest'
            for spec in self.names:
                if is_newer_version_installed(base, spec):
                    if self.allow_downgrade:
                        if upgrade:
                            if is_installed(base, spec):
                                goal.add_upgrade(spec, settings)
                            else:
                                goal.add_install(spec, settings)
                        else:
                            goal.add_install(spec, settings)
                elif is_installed(base, spec):
                    if upgrade:
                        goal.add_upgrade(spec, settings)
                elif self.update_only:
                    results.append('Packages providing {} not installed due to update_only specified'.format(spec))
                else:
                    goal.add_install(spec, settings)
        elif self.state in {'absent', 'removed'}:
            for spec in self.names:
                try:
                    goal.add_remove(spec, settings)
                except RuntimeError as e:
                    self.module.fail_json(msg=str(e), failures=[], rc=1)
            if self.autoremove:
                for pkg in get_unneeded_pkgs(base):
                    goal.add_rpm_remove(pkg, settings)
        goal.set_allow_erasing(self.allowerasing)
        try:
            transaction = goal.resolve()
        except RuntimeError as e:
            self.module.fail_json(msg=str(e), failures=[], rc=1)
        if transaction.get_problems():
            failures = []
            for log_event in transaction.get_resolve_logs():
                if log_event.get_problem() == libdnf5.base.GoalProblem_NOT_FOUND and self.state in {'install', 'present', 'latest'}:
                    failures.append('No package {} available.'.format(log_event.get_spec()))
                else:
                    failures.append(log_event.to_string())
            if transaction.get_problems() & libdnf5.base.GoalProblem_SOLVER_ERROR != 0:
                msg = 'Depsolve Error occurred'
            else:
                msg = 'Failed to install some of the specified packages'
            self.module.fail_json(msg=msg, failures=failures, rc=1)
        actions_compat_map = {'Install': 'Installed', 'Remove': 'Removed', 'Replace': 'Installed', 'Upgrade': 'Installed', 'Replaced': 'Removed'}
        changed = bool(transaction.get_transaction_packages())
        for pkg in transaction.get_transaction_packages():
            if self.download_only:
                action = 'Downloaded'
            else:
                action = libdnf5.base.transaction.transaction_item_action_to_string(pkg.get_action())
            results.append('{}: {}'.format(actions_compat_map.get(action, action), pkg.get_package().get_nevra()))
        msg = ''
        if self.module.check_mode:
            if results:
                msg = 'Check mode: No changes made, but would have if not in check mode'
        else:
            transaction.download()
            if not self.download_only:
                transaction.set_description('ansible dnf5 module')
                result = transaction.run()
                if result == libdnf5.base.Transaction.TransactionRunResult_ERROR_GPG_CHECK:
                    self.module.fail_json(msg='Failed to validate GPG signatures: {}'.format(','.join(transaction.get_gpg_signature_problems())), failures=[], rc=1)
                elif result != libdnf5.base.Transaction.TransactionRunResult_SUCCESS:
                    self.module.fail_json(msg='Failed to install some of the specified packages', failures=['{}: {}'.format(transaction.transaction_result_to_string(result), log) for log in transaction.get_transaction_problems()], rc=1)
        if not msg and (not results):
            msg = 'Nothing to do'
        self.module.exit_json(results=results, changed=changed, msg=msg, rc=0)