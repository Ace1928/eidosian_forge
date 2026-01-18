import contextlib
import errno
import hashlib
import itertools
import json
import logging
import os
import os.path as osp
import re
import shutil
import site
import stat
import subprocess
import sys
import tarfile
from copy import deepcopy
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import FrozenSet, Optional
from urllib.error import URLError
from urllib.request import Request, quote, urljoin, urlopen
from jupyter_core.paths import jupyter_config_dir
from jupyter_server.extension.serverextension import GREEN_ENABLED, GREEN_OK, RED_DISABLED, RED_X
from jupyterlab_server.config import (
from jupyterlab_server.process import Process, WatchHelper, list2cmdline, which
from packaging.version import Version
from traitlets import Bool, HasTraits, Instance, List, Unicode, default
from jupyterlab._version import __version__
from jupyterlab.coreconfig import CoreConfig
from jupyterlab.jlpmapp import HERE, YARN_PATH
from jupyterlab.semver import Range, gt, gte, lt, lte, make_semver
class _AppHandler:

    def __init__(self, options):
        """Create a new _AppHandler object"""
        options = _ensure_options(options)
        self._options = options
        self.app_dir = options.app_dir
        self.sys_dir = get_app_dir() if options.use_sys_dir else self.app_dir
        self.logger = options.logger
        self.core_data = deepcopy(options.core_config._data)
        self.labextensions_path = options.labextensions_path
        self.kill_event = options.kill_event
        self.registry = options.registry
        self.skip_full_build_check = options.skip_full_build_check
        self.info = self._get_app_info()
        try:
            self._maybe_mirror_disabled_in_locked(level='sys_prefix')
        except (PermissionError, OSError):
            try:
                self.logger.info('`sys_prefix` level settings are read-only, using `user` level for migration to `lockedExtensions`')
                self._maybe_mirror_disabled_in_locked(level='user')
            except (PermissionError, OSError):
                self.logger.warning('Both `sys_prefix` and `user` level settings are read-only, cannot auto-migrate `disabledExtensions` to `lockedExtensions`')

    def install_extension(self, extension, existing=None, pin=None):
        """Install an extension package into JupyterLab.

        The extension is first validated.

        Returns `True` if a rebuild is recommended, `False` otherwise.
        """
        extension = _normalize_path(extension)
        extensions = self.info['extensions']
        if extension in self.info['core_extensions']:
            config = self._read_build_config()
            uninstalled = config.get('uninstalled_core_extensions', [])
            if extension in uninstalled:
                self.logger.info('Installing core extension %s' % extension)
                uninstalled.remove(extension)
                config['uninstalled_core_extensions'] = uninstalled
                self._write_build_config(config)
                return True
            return False
        self._ensure_app_dirs()
        with TemporaryDirectory() as tempdir:
            info = self._install_extension(extension, tempdir, pin=pin)
        name = info['name']
        if info['is_dir']:
            config = self._read_build_config()
            local = config.setdefault('local_extensions', {})
            local[name] = info['source']
            self._write_build_config(config)
        if name in extensions:
            other = extensions[name]
            if other['path'] != info['path'] and other['location'] == 'app':
                os.remove(other['path'])
        return True

    def build(self, name=None, version=None, static_url=None, clean_staging=False, production=True, minimize=True):
        """Build the application."""
        if production is None:
            production = not (self.info['linked_packages'] or self.info['local_extensions'])
        if not production:
            minimize = False
        if self._options.splice_source:
            ensure_node_modules(REPO_ROOT, logger=self.logger)
            self._run(['node', YARN_PATH, 'build:packages'], cwd=REPO_ROOT)
        info = ['production' if production else 'development']
        if production:
            info.append('minimized' if minimize else 'not minimized')
        self.logger.info(f'Building jupyterlab assets ({', '.join(info)})')
        app_dir = self.app_dir
        self._populate_staging(name=name, version=version, static_url=static_url, clean=clean_staging)
        staging = pjoin(app_dir, 'staging')
        ret = self._run(['node', YARN_PATH, 'install'], cwd=staging)
        if ret != 0:
            msg = 'npm dependencies failed to install'
            self.logger.debug(msg)
            raise RuntimeError(msg)
        dedupe_yarn(staging, self.logger)
        command = f'build:{('prod' if production else 'dev')}{(':minimize' if minimize else '')}'
        ret = self._run(['node', YARN_PATH, 'run', command], cwd=staging)
        if ret != 0:
            msg = 'JupyterLab failed to build'
            self.logger.debug(msg)
            raise RuntimeError(msg)

    def watch(self):
        """Start the application watcher and then run the watch in
        the background.
        """
        staging = pjoin(self.app_dir, 'staging')
        self._populate_staging()
        self._run(['node', YARN_PATH, 'install'], cwd=staging)
        dedupe_yarn(staging, self.logger)
        proc = WatchHelper(['node', YARN_PATH, 'run', 'watch'], cwd=pjoin(self.app_dir, 'staging'), startup_regex=WEBPACK_EXPECT, logger=self.logger)
        return [proc]

    def list_extensions(self):
        """Print an output of the extensions."""
        self._ensure_disabled_info()
        logger = self.logger
        info = self.info
        logger.info('JupyterLab v%s' % info['version'])
        if info['federated_extensions'] or info['extensions']:
            info['compat_errors'] = self._get_extension_compat()
        if info['federated_extensions']:
            self._list_federated_extensions()
        if info['extensions']:
            logger.info('Other labextensions (built into JupyterLab)')
            self._list_extensions(info, 'app')
            self._list_extensions(info, 'sys')
        local = info['local_extensions']
        if local:
            logger.info('\n   local extensions:')
            for name in sorted(local):
                logger.info(f'        {name}: {local[name]}')
        linked_packages = info['linked_packages']
        if linked_packages:
            logger.info('\n   linked packages:')
            for key in sorted(linked_packages):
                source = linked_packages[key]['source']
                logger.info(f'        {key}: {source}')
        uninstalled_core = info['uninstalled_core']
        if uninstalled_core:
            logger.info('\nUninstalled core extensions:')
            [logger.info('    %s' % item) for item in sorted(uninstalled_core)]
        all_exts = list(info['federated_extensions']) + list(info['extensions']) + list(info['core_extensions'])
        disabled = [i for i in info['disabled'] if i.partition(':')[0] in all_exts]
        if disabled:
            logger.info('\nDisabled extensions:')
            for item in sorted(disabled):
                if item in all_exts:
                    item += ' (all plugins)'
                logger.info('    %s' % item)
        improper_shadowed = []
        for ext_name in self.info['shadowed_exts']:
            source_version = self.info['extensions'][ext_name]['version']
            prebuilt_version = self.info['federated_extensions'][ext_name]['version']
            if not gte(prebuilt_version, source_version, True):
                improper_shadowed.append(ext_name)
        if improper_shadowed:
            logger.info('\nThe following source extensions are overshadowed by older prebuilt extensions:')
            [logger.info('    %s' % name) for name in sorted(improper_shadowed)]
        messages = self.build_check(fast=True)
        if messages:
            logger.info('\nBuild recommended, please run `jupyter lab build`:')
            [logger.info('    %s' % item) for item in messages]

    def build_check(self, fast=None):
        """Determine whether JupyterLab should be built.

        Returns a list of messages.
        """
        if fast is None:
            fast = self.skip_full_build_check
        app_dir = self.app_dir
        local = self.info['local_extensions']
        linked = self.info['linked_packages']
        messages = []
        pkg_path = pjoin(app_dir, 'static', 'package.json')
        if not osp.exists(pkg_path):
            return ['No built application']
        static_data = self.info['static_data']
        old_jlab = static_data['jupyterlab']
        old_deps = static_data.get('dependencies', {})
        static_version = old_jlab.get('version', '')
        if not static_version.endswith('-spliced'):
            core_version = old_jlab['version']
            if Version(static_version) != Version(core_version):
                msg = 'Version mismatch: %s (built), %s (current)'
                return [msg % (static_version, core_version)]
        shadowed_exts = self.info['shadowed_exts']
        new_package = self._get_package_template(silent=fast)
        new_jlab = new_package['jupyterlab']
        new_deps = new_package.get('dependencies', {})
        for ext_type in ['extensions', 'mimeExtensions']:
            for ext in new_jlab[ext_type]:
                if ext in shadowed_exts:
                    continue
                if ext not in old_jlab[ext_type]:
                    messages.append('%s needs to be included in build' % ext)
            for ext in old_jlab[ext_type]:
                if ext in shadowed_exts:
                    continue
                if ext not in new_jlab[ext_type]:
                    messages.append('%s needs to be removed from build' % ext)
        src_pkg_dir = pjoin(REPO_ROOT, 'packages')
        for pkg, dep in new_deps.items():
            if old_deps.get(pkg, '').startswith(src_pkg_dir):
                continue
            if pkg not in old_deps:
                continue
            if pkg in local or pkg in linked:
                continue
            if old_deps[pkg] != dep:
                msg = '%s changed from %s to %s'
                messages.append(msg % (pkg, old_deps[pkg], new_deps[pkg]))
        for name, source in local.items():
            if fast or name in shadowed_exts:
                continue
            dname = pjoin(app_dir, 'extensions')
            if self._check_local(name, source, dname):
                messages.append('%s content changed' % name)
        for name, item in linked.items():
            if fast or name in shadowed_exts:
                continue
            dname = pjoin(app_dir, 'staging', 'linked_packages')
            if self._check_local(name, item['source'], dname):
                messages.append('%s content changed' % name)
        return messages

    def uninstall_extension(self, name):
        """Uninstall an extension by name.

        Returns `True` if a rebuild is recommended, `False` otherwise.
        """
        info = self.info
        logger = self.logger
        if name in info['federated_extensions']:
            if info['federated_extensions'][name].get('install', {}).get('uninstallInstructions', None):
                logger.error('JupyterLab cannot uninstall this extension. %s' % info['federated_extensions'][name]['install']['uninstallInstructions'])
            else:
                logger.error('JupyterLab cannot uninstall %s since it was installed outside of JupyterLab. Use the same method used to install this extension to uninstall this extension.' % name)
            return False
        if name in info['core_extensions']:
            config = self._read_build_config()
            uninstalled = config.get('uninstalled_core_extensions', [])
            if name not in uninstalled:
                logger.info('Uninstalling core extension %s' % name)
                uninstalled.append(name)
                config['uninstalled_core_extensions'] = uninstalled
                self._write_build_config(config)
                return True
            return False
        local = info['local_extensions']
        for extname, data in info['extensions'].items():
            path = data['path']
            if extname == name:
                msg = f'Uninstalling {name} from {osp.dirname(path)}'
                logger.info(msg)
                os.remove(path)
                if extname in local:
                    config = self._read_build_config()
                    data = config.setdefault('local_extensions', {})
                    del data[extname]
                    self._write_build_config(config)
                return True
        logger.warning('No labextension named "%s" installed' % name)
        return False

    def uninstall_all_extensions(self):
        """Uninstalls all extensions

        Returns `True` if a rebuild is recommended, `False` otherwise
        """
        should_rebuild = False
        for extname, _ in self.info['extensions'].items():
            uninstalled = self.uninstall_extension(extname)
            should_rebuild = should_rebuild or uninstalled
        return should_rebuild

    def update_all_extensions(self):
        """Update all non-local extensions.

        Returns `True` if a rebuild is recommended, `False` otherwise.
        """
        should_rebuild = False
        for extname, _ in self.info['extensions'].items():
            if extname in self.info['local_extensions']:
                continue
            updated = self._update_extension(extname)
            should_rebuild = should_rebuild or updated
        return should_rebuild

    def update_extension(self, name):
        """Update an extension by name.

        Returns `True` if a rebuild is recommended, `False` otherwise.
        """
        if name not in self.info['extensions']:
            self.logger.warning('No labextension named "%s" installed' % name)
            return False
        return self._update_extension(name)

    def _update_extension(self, name):
        """Update an extension by name.

        Returns `True` if a rebuild is recommended, `False` otherwise.
        """
        data = self.info['extensions'][name]
        if data['alias_package_source']:
            self.logger.warning("Skipping updating pinned extension '%s'." % name)
            return False
        try:
            latest = self._latest_compatible_package_version(name)
        except URLError:
            return False
        if latest is None:
            self.logger.warning(f'No compatible version found for {name}!')
            return False
        if latest == data['version']:
            self.logger.info('Extension %r already up to date' % name)
            return False
        self.logger.info(f'Updating {name} to version {latest}')
        return self.install_extension(f'{name}@{latest}')

    def link_package(self, path):
        """Link a package at the given path.

        Returns `True` if a rebuild is recommended, `False` otherwise.
        """
        path = _normalize_path(path)
        if not osp.exists(path) or not osp.isdir(path):
            msg = 'Cannot install "%s" only link local directories'
            raise ValueError(msg % path)
        with TemporaryDirectory() as tempdir:
            info = self._extract_package(path, tempdir)
        messages = _validate_extension(info['data'])
        if not messages:
            return self.install_extension(path)
        self.logger.warning('Installing %s as a linked package because it does not have extension metadata:', path)
        [self.logger.warning('   %s' % m) for m in messages]
        config = self._read_build_config()
        linked = config.setdefault('linked_packages', {})
        linked[info['name']] = info['source']
        self._write_build_config(config)
        return True

    def unlink_package(self, path):
        """Unlink a package by name or at the given path.

        A ValueError is raised if the path is not an unlinkable package.

        Returns `True` if a rebuild is recommended, `False` otherwise.
        """
        path = _normalize_path(path)
        config = self._read_build_config()
        linked = config.setdefault('linked_packages', {})
        found = None
        for name, source in linked.items():
            if path in {name, source}:
                found = name
        if found:
            del linked[found]
        else:
            local = config.setdefault('local_extensions', {})
            for name, source in local.items():
                if path in {name, source}:
                    found = name
            if found:
                del local[found]
                path = self.info['extensions'][found]['path']
                os.remove(path)
        if not found:
            raise ValueError('No linked package for %s' % path)
        self._write_build_config(config)
        return True

    def toggle_extension(self, extension, value, level='sys_prefix'):
        """Enable or disable a lab extension.

        Returns `True` if a rebuild is recommended, `False` otherwise.
        """
        app_settings_dir = osp.join(self.app_dir, 'settings')
        page_config = get_static_page_config(app_settings_dir=app_settings_dir, logger=self.logger, level=level)
        disabled = page_config.get('disabledExtensions', {})
        did_something = False
        is_disabled = disabled.get(extension, False)
        if value and (not is_disabled):
            disabled[extension] = True
            did_something = True
        elif not value and is_disabled:
            disabled[extension] = False
            did_something = True
        if did_something:
            page_config['disabledExtensions'] = disabled
            write_page_config(page_config, level=level)
        return did_something

    def _maybe_mirror_disabled_in_locked(self, level='sys_prefix'):
        """Lock all extensions that were previously disabled.

        This exists to facilitate migration from 4.0 (which did not include lock
        function) to 4.1 which exposes the plugin management to users in UI.

        Returns `True` if migration happened, `False` otherwise.
        """
        app_settings_dir = osp.join(self.app_dir, 'settings')
        page_config = get_static_page_config(app_settings_dir=app_settings_dir, logger=self.logger, level=level)
        if 'lockedExtensions' in page_config:
            return False
        disabled = page_config.get('disabledExtensions', {})
        if isinstance(disabled, list):
            disabled = {extension: True for extension in disabled}
        page_config['lockedExtensions'] = disabled
        write_page_config(page_config, level=level)
        return True

    def toggle_extension_lock(self, extension, value, level='sys_prefix'):
        """Lock or unlock a lab extension (/plugin)."""
        app_settings_dir = osp.join(self.app_dir, 'settings')
        page_config = get_static_page_config(app_settings_dir=app_settings_dir, logger=self.logger, level=level)
        locked = page_config.get('lockedExtensions', {})
        locked[extension] = value
        write_page_config(page_config, level=level)

    def check_extension(self, extension, check_installed_only=False):
        """Check if a lab extension is enabled or disabled"""
        self._ensure_disabled_info()
        info = self.info
        if extension in info['core_extensions']:
            return self._check_core_extension(extension, info, check_installed_only)
        if extension in info['linked_packages']:
            self.logger.info(f'{extension}:{GREEN_ENABLED}')
            return True
        return self._check_common_extension(extension, info, check_installed_only)

    def _check_core_extension(self, extension, info, check_installed_only):
        """Check if a core extension is enabled or disabled"""
        if extension in info['uninstalled_core']:
            self.logger.info(f'{extension}:{RED_X}')
            return False
        if check_installed_only:
            self.logger.info(f'{extension}: {GREEN_OK}')
            return True
        if extension in info['disabled_core']:
            self.logger.info(f'{extension}: {RED_DISABLED}')
            return False
        self.logger.info(f'{extension}:{GREEN_ENABLED}')
        return True

    def _check_common_extension(self, extension, info, check_installed_only):
        """Check if a common (non-core) extension is enabled or disabled"""
        if extension not in info['extensions']:
            self.logger.info(f'{extension}:{RED_X}')
            return False
        errors = self._get_extension_compat()[extension]
        if errors:
            self.logger.info(f'{extension}:{RED_X} (compatibility errors)')
            return False
        if check_installed_only:
            self.logger.info(f'{extension}: {GREEN_OK}')
            return True
        if _is_disabled(extension, info['disabled']):
            self.logger.info(f'{extension}: {RED_DISABLED}')
            return False
        self.logger.info(f'{extension}:{GREEN_ENABLED}')
        return True

    def _get_app_info(self):
        """Get information about the app."""
        info = {}
        info['core_data'] = core_data = self.core_data
        info['extensions'] = extensions = self._get_extensions(core_data)
        info['local_extensions'] = self._get_local_extensions()
        info['linked_packages'] = self._get_linked_packages()
        info['app_extensions'] = app = []
        info['sys_extensions'] = sys = []
        for name, data in extensions.items():
            data['is_local'] = name in info['local_extensions']
            if data['location'] == 'app':
                app.append(name)
            else:
                sys.append(name)
        info['uninstalled_core'] = self._get_uninstalled_core_extensions()
        info['static_data'] = _get_static_data(self.app_dir)
        app_data = info['static_data'] or core_data
        info['version'] = app_data['jupyterlab']['version']
        info['staticUrl'] = app_data['jupyterlab'].get('staticUrl', '')
        info['sys_dir'] = self.sys_dir
        info['app_dir'] = self.app_dir
        info['core_extensions'] = _get_core_extensions(self.core_data)
        info['federated_extensions'] = get_federated_extensions(self.labextensions_path)
        info['shadowed_exts'] = [ext for ext in info['extensions'] if ext in info['federated_extensions']]
        return info

    def _ensure_disabled_info(self):
        info = self.info
        if 'disabled' in info:
            return
        labextensions_path = self.labextensions_path
        app_settings_dir = osp.join(self.app_dir, 'settings')
        page_config = get_page_config(labextensions_path, app_settings_dir=app_settings_dir, logger=self.logger)
        disabled = page_config.get('disabledExtensions', {})
        if isinstance(disabled, list):
            disabled = {extension: True for extension in disabled}
        info['disabled'] = disabled
        locked = page_config.get('lockedExtensions', {})
        if isinstance(locked, list):
            locked = {extension: True for extension in locked}
        info['locked'] = locked
        disabled_core = []
        for key in info['core_extensions']:
            if key in info['disabled']:
                disabled_core.append(key)
        info['disabled_core'] = disabled_core

    def _populate_staging(self, name=None, version=None, static_url=None, clean=False):
        """Set up the assets in the staging directory."""
        app_dir = self.app_dir
        staging = pjoin(app_dir, 'staging')
        if clean and osp.exists(staging):
            self.logger.info('Cleaning %s', staging)
            _rmtree(staging, self.logger)
        self._ensure_app_dirs()
        if not version:
            version = self.info['core_data']['jupyterlab']['version']
        splice_source = self._options.splice_source
        if splice_source:
            self.logger.debug('Splicing dev packages into app directory.')
            source_dir = DEV_DIR
            version = __version__ + '-spliced'
        else:
            source_dir = pjoin(HERE, 'staging')
        pkg_path = pjoin(staging, 'package.json')
        if osp.exists(pkg_path):
            with open(pkg_path) as fid:
                data = json.load(fid)
            if data['jupyterlab'].get('version', '') != version:
                _rmtree(staging, self.logger)
                os.makedirs(staging)
        for fname in ['index.js', 'bootstrap.js', 'publicpath.js', 'webpack.config.js', 'webpack.prod.config.js', 'webpack.prod.minimize.config.js']:
            target = pjoin(staging, fname)
            shutil.copy(pjoin(source_dir, fname), target)
        for fname in ['.yarnrc.yml', 'yarn.js']:
            target = pjoin(staging, fname)
            shutil.copy(pjoin(HERE, 'staging', fname), target)
        templates = pjoin(staging, 'templates')
        if osp.exists(templates):
            _rmtree(templates, self.logger)
        try:
            shutil.copytree(pjoin(source_dir, 'templates'), templates)
        except shutil.Error as error:
            real_error = '[Errno 22]' not in str(error) and '[Errno 5]' not in str(error)
            if real_error or not osp.exists(templates):
                raise
        linked_dir = pjoin(staging, 'linked_packages')
        if osp.exists(linked_dir):
            _rmtree(linked_dir, self.logger)
        os.makedirs(linked_dir)
        extensions = self.info['extensions']
        removed = False
        for key, source in self.info['local_extensions'].items():
            if key not in extensions:
                config = self._read_build_config()
                data = config.setdefault('local_extensions', {})
                del data[key]
                self._write_build_config(config)
                removed = True
                continue
            dname = pjoin(app_dir, 'extensions')
            self._update_local(key, source, dname, extensions[key], 'local_extensions')
        if removed:
            self.info['local_extensions'] = self._get_local_extensions()
        linked = self.info['linked_packages']
        for key, item in linked.items():
            dname = pjoin(staging, 'linked_packages')
            self._update_local(key, item['source'], dname, item, 'linked_packages')
        data = self._get_package_template()
        jlab = data['jupyterlab']
        if version:
            jlab['version'] = version
        if name:
            jlab['name'] = name
        if static_url:
            jlab['staticUrl'] = static_url
        if splice_source:
            for path in glob(pjoin(REPO_ROOT, 'packages', '*', 'package.json')):
                local_path = osp.dirname(osp.abspath(path))
                pkg_data = json.loads(Path(path).read_text(encoding='utf-8'))
                name = pkg_data['name']
                if name in data['dependencies']:
                    data['dependencies'][name] = local_path
                    jlab['linkedPackages'][name] = local_path
                if name in data['resolutions']:
                    data['resolutions'][name] = local_path
            local_path = osp.abspath(pjoin(REPO_ROOT, 'builder'))
            data['devDependencies']['@jupyterlab/builder'] = local_path
            target = osp.join(staging, 'node_modules', '@jupyterlab', 'builder')
            node_modules = pjoin(staging, 'node_modules')
            if osp.exists(node_modules):
                shutil.rmtree(node_modules, ignore_errors=True)
        pkg_path = pjoin(staging, 'package.json')
        with open(pkg_path, 'w') as fid:
            json.dump(data, fid, indent=4)
        lock_path = pjoin(staging, 'yarn.lock')
        lock_template = pjoin(HERE, 'staging', 'yarn.lock')
        if not osp.exists(lock_path):
            shutil.copy(lock_template, lock_path)
            os.chmod(lock_path, stat.S_IWRITE | stat.S_IREAD)

    def _get_package_template(self, silent=False):
        """Get the template the for staging package.json file."""
        logger = self.logger
        data = deepcopy(self.info['core_data'])
        local = self.info['local_extensions']
        linked = self.info['linked_packages']
        extensions = self.info['extensions']
        shadowed_exts = self.info['shadowed_exts']
        jlab = data['jupyterlab']

        def format_path(path):
            path = osp.relpath(path, osp.abspath(osp.realpath(pjoin(self.app_dir, 'staging'))))
            path = 'file:' + path.replace(os.sep, '/')
            if os.name == 'nt':
                path = path.lower()
            return path
        jlab['linkedPackages'] = {}
        for key, source in local.items():
            if key in shadowed_exts:
                continue
            jlab['linkedPackages'][key] = source
            data['resolutions'][key] = 'file:' + self.info['extensions'][key]['path']
        for key, item in linked.items():
            if key in shadowed_exts:
                continue
            path = pjoin(self.app_dir, 'staging', 'linked_packages')
            path = pjoin(path, item['filename'])
            data['dependencies'][key] = format_path(path)
            jlab['linkedPackages'][key] = item['source']
            data['resolutions'][key] = format_path(path)
        data['jupyterlab']['extensionMetadata'] = {}
        compat_errors = self._get_extension_compat()
        for key, value in extensions.items():
            errors = compat_errors[key]
            if errors:
                if not silent:
                    _log_single_compat_errors(logger, key, value['version'], errors)
                continue
            data['dependencies'][key] = format_path(value['path'])
            jlab_data = value['jupyterlab']
            for item in ['extension', 'mimeExtension']:
                ext = jlab_data.get(item, False)
                if not ext:
                    continue
                if ext is True:
                    ext = ''
                jlab[item + 's'][key] = ext
                data['jupyterlab']['extensionMetadata'][key] = jlab_data
        for item in self.info['uninstalled_core']:
            if item in jlab['extensions']:
                data['jupyterlab']['extensions'].pop(item)
            elif item in jlab['mimeExtensions']:
                data['jupyterlab']['mimeExtensions'].pop(item)
            if item in data['dependencies']:
                data['dependencies'].pop(item)
        return data

    def _check_local(self, name, source, dname):
        """Check if a local package has changed.

        `dname` is the directory name of existing package tar archives.
        """
        with TemporaryDirectory() as tempdir:
            info = self._extract_package(source, tempdir)
            target = pjoin(dname, info['filename'])
            return not osp.exists(target)

    def _update_local(self, name, source, dname, data, dtype):
        """Update a local dependency.  Return `True` if changed."""
        existing = data['filename']
        if not osp.exists(pjoin(dname, existing)):
            existing = ''
        with TemporaryDirectory() as tempdir:
            info = self._extract_package(source, tempdir)
            if info['filename'] == existing:
                return existing
            shutil.move(info['path'], pjoin(dname, info['filename']))
        if existing:
            os.remove(pjoin(dname, existing))
        data['filename'] = info['filename']
        data['path'] = pjoin(data['tar_dir'], data['filename'])
        return info['filename']

    def _get_extensions(self, core_data):
        """Get the extensions for the application."""
        app_dir = self.app_dir
        extensions = {}
        sys_path = pjoin(self.sys_dir, 'extensions')
        app_path = pjoin(self.app_dir, 'extensions')
        extensions = self._get_extensions_in_dir(self.sys_dir, core_data)
        app_path = pjoin(app_dir, 'extensions')
        if app_path == sys_path or not osp.exists(app_path):
            return extensions
        extensions.update(self._get_extensions_in_dir(app_dir, core_data))
        return extensions

    def _get_extensions_in_dir(self, dname, core_data):
        """Get the extensions in a given directory."""
        extensions = {}
        location = 'app' if dname == self.app_dir else 'sys'
        for target in glob(pjoin(dname, 'extensions', '*.tgz')):
            data = read_package(target)
            deps = data.get('dependencies', {})
            name = data['name']
            jlab = data.get('jupyterlab', {})
            path = osp.abspath(target)
            filename = osp.basename(target)
            if filename.startswith(PIN_PREFIX):
                alias = filename[len(PIN_PREFIX):-len('.tgz')]
            else:
                alias = None
            url = get_package_url(data)
            extensions[alias or name] = {'description': data.get('description', ''), 'path': path, 'filename': osp.basename(path), 'url': url, 'version': data['version'], 'alias_package_source': name if alias else None, 'jupyterlab': jlab, 'dependencies': deps, 'tar_dir': osp.dirname(path), 'location': location}
        return extensions

    def _get_extension_compat(self):
        """Get the extension compatibility info."""
        compat = {}
        core_data = self.info['core_data']
        seen = set()
        for name, data in self.info['federated_extensions'].items():
            deps = data['dependencies']
            compat[name] = _validate_compatibility(name, deps, core_data)
            seen.add(name)
        for name, data in self.info['extensions'].items():
            if name in seen:
                continue
            deps = data['dependencies']
            compat[name] = _validate_compatibility(name, deps, core_data)
        return compat

    def _get_local_extensions(self):
        """Get the locally installed extensions."""
        return self._get_local_data('local_extensions')

    def _get_linked_packages(self):
        """Get the linked packages."""
        info = self._get_local_data('linked_packages')
        dname = pjoin(self.app_dir, 'staging', 'linked_packages')
        for name, source in info.items():
            info[name] = {'source': source, 'filename': '', 'tar_dir': dname}
        if not osp.exists(dname):
            return info
        for path in glob(pjoin(dname, '*.tgz')):
            path = osp.abspath(path)
            data = read_package(path)
            name = data['name']
            if name not in info:
                self.logger.warning('Removing orphaned linked package %s' % name)
                os.remove(path)
                continue
            item = info[name]
            item['filename'] = osp.basename(path)
            item['path'] = path
            item['version'] = data['version']
            item['data'] = data
        return info

    def _get_uninstalled_core_extensions(self):
        """Get the uninstalled core extensions."""
        config = self._read_build_config()
        return config.get('uninstalled_core_extensions', [])

    def _ensure_app_dirs(self):
        """Ensure that the application directories exist"""
        dirs = ['extensions', 'settings', 'staging', 'schemas', 'themes']
        for dname in dirs:
            path = pjoin(self.app_dir, dname)
            if not osp.exists(path):
                try:
                    os.makedirs(path)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise

    def _list_extensions(self, info, ext_type):
        """List the extensions of a given type."""
        self._ensure_disabled_info()
        logger = self.logger
        names = info['%s_extensions' % ext_type]
        if not names:
            return
        dname = info['%s_dir' % ext_type]
        error_accumulator = {}
        logger.info(f'   {ext_type} dir: {dname}')
        for name in sorted(names):
            if name in info['federated_extensions']:
                continue
            data = info['extensions'][name]
            version = data['version']
            errors = info['compat_errors'][name]
            extra = self._compose_extra_status(name, info, data, errors)
            alias_package_source = data['alias_package_source']
            if alias_package_source:
                logger.info(f'        {name} {alias_package_source} v{version}{extra}')
            else:
                logger.info(f'        {name} v{version}{extra}')
            if errors:
                error_accumulator[name] = (version, errors)
        _log_multiple_compat_errors(logger, error_accumulator)
        logger.info('')

    def _list_federated_extensions(self):
        self._ensure_disabled_info()
        info = self.info
        logger = self.logger
        error_accumulator = {}
        ext_dirs = {p: False for p in self.labextensions_path}
        for value in info['federated_extensions'].values():
            ext_dirs[value['ext_dir']] = True
        for ext_dir, has_exts in ext_dirs.items():
            if not has_exts:
                continue
            logger.info(ext_dir)
            for name in info['federated_extensions']:
                data = info['federated_extensions'][name]
                if data['ext_dir'] != ext_dir:
                    continue
                version = data['version']
                errors = info['compat_errors'][name]
                extra = self._compose_extra_status(name, info, data, errors)
                install = data.get('install')
                if install:
                    extra += ' ({}, {})'.format(install['packageManager'], install['packageName'])
                logger.info(f'        {name} v{version}{extra}')
                if errors:
                    error_accumulator[name] = (version, errors)
            logger.info('')
        _log_multiple_compat_errors(logger, error_accumulator)

    def _compose_extra_status(self, name: str, info: dict, data: dict, errors) -> str:
        extra = ''
        if _is_disabled(name, info['disabled']):
            extra += ' %s' % RED_DISABLED
        else:
            extra += ' %s' % GREEN_ENABLED
        if errors:
            extra += ' %s' % RED_X
        else:
            extra += ' %s' % GREEN_OK
        if data['is_local']:
            extra += '*'
        lock_status = _is_locked(name, info['locked'])
        if lock_status.entire_extension_locked:
            extra += ' 🔒 (all plugins locked)'
        elif lock_status.locked_plugins:
            plugin_list = ', '.join(sorted(lock_status.locked_plugins))
            extra += ' 🔒 (plugins: %s locked)' % plugin_list
        return extra

    def _read_build_config(self):
        """Get the build config data for the app dir."""
        target = pjoin(self.app_dir, 'settings', 'build_config.json')
        if not osp.exists(target):
            return {}
        else:
            with open(target) as fid:
                return json.load(fid)

    def _write_build_config(self, config):
        """Write the build config to the app dir."""
        self._ensure_app_dirs()
        target = pjoin(self.app_dir, 'settings', 'build_config.json')
        with open(target, 'w') as fid:
            json.dump(config, fid, indent=4)

    def _get_local_data(self, source):
        """Get the local data for extensions or linked packages."""
        config = self._read_build_config()
        data = config.setdefault(source, {})
        dead = []
        for name, source in data.items():
            if not osp.exists(source):
                dead.append(name)
        for name in dead:
            link_type = source.replace('_', ' ')
            msg = f'**Note: Removing dead {link_type} "{name}"'
            self.logger.warning(msg)
            del data[name]
        if dead:
            self._write_build_config(config)
        return data

    def _install_extension(self, extension, tempdir, pin=None):
        """Install an extension with validation and return the name and path."""
        info = self._extract_package(extension, tempdir, pin=pin)
        data = info['data']
        allow_fallback = '@' not in extension[1:] and (not info['is_dir'])
        name = info['name']
        messages = _validate_extension(data)
        if messages:
            msg = '"%s" is not a valid extension:\n%s'
            msg = msg % (extension, '\n'.join(messages))
            if allow_fallback:
                try:
                    version = self._latest_compatible_package_version(name)
                except URLError:
                    raise ValueError(msg) from None
            else:
                raise ValueError(msg)
        deps = data.get('dependencies', {})
        errors = _validate_compatibility(extension, deps, self.core_data)
        if errors:
            msg = _format_compatibility_errors(data['name'], data['version'], errors)
            if allow_fallback:
                try:
                    version = self._latest_compatible_package_version(name)
                except URLError:
                    raise ValueError(msg) from None
                if version and name:
                    self.logger.debug('Incompatible extension:\n%s', name)
                    self.logger.debug('Found compatible version: %s', version)
                    with TemporaryDirectory() as tempdir2:
                        return self._install_extension(f'{name}@{version}', tempdir2)
                conflicts = '\n'.join(msg.splitlines()[2:])
                msg = ''.join((self._format_no_compatible_package_version(name), '\n\n', conflicts))
            raise ValueError(msg)
        target = pjoin(self.app_dir, 'extensions', info['filename'])
        if osp.exists(target):
            os.remove(target)
        shutil.move(info['path'], target)
        info['path'] = target
        return info

    def _extract_package(self, source, tempdir, pin=None):
        """Call `npm pack` for an extension.

        The pack command will download the package tar if `source` is
        a package name, or run `npm pack` locally if `source` is a
        directory.
        """
        is_dir = osp.exists(source) and osp.isdir(source)
        if is_dir and (not osp.exists(pjoin(source, 'node_modules'))):
            self._run(['node', YARN_PATH, 'install'], cwd=source)
        info = {'source': source, 'is_dir': is_dir}
        ret = self._run([which('npm'), 'pack', source], cwd=tempdir)
        if ret != 0:
            msg = '"%s" is not a valid npm package'
            raise ValueError(msg % source)
        path = glob(pjoin(tempdir, '*.tgz'))[0]
        info['data'] = read_package(path)
        if is_dir:
            info['sha'] = sha = _tarsum(path)
            target = path.replace('.tgz', '-%s.tgz' % sha)
            shutil.move(path, target)
            info['path'] = target
        else:
            info['path'] = path
        if pin:
            old_path = info['path']
            new_path = pjoin(osp.dirname(old_path), f'{PIN_PREFIX}{pin}.tgz')
            shutil.move(old_path, new_path)
            info['path'] = new_path
        info['filename'] = osp.basename(info['path'])
        info['name'] = info['data']['name']
        info['version'] = info['data']['version']
        return info

    def _latest_compatible_package_version(self, name):
        """Get the latest compatible version of a package"""
        core_data = self.info['core_data']
        try:
            metadata = _fetch_package_metadata(self.registry, name, self.logger)
        except URLError:
            return
        versions = metadata.get('versions', {})

        def sort_key(key_value):
            return _semver_key(key_value[0], prerelease_first=True)
        for version, data in sorted(versions.items(), key=sort_key, reverse=True):
            deps = data.get('dependencies', {})
            errors = _validate_compatibility(name, deps, core_data)
            if not errors:
                if 'deprecated' in data:
                    self.logger.debug(f'Disregarding compatible version of package as it is deprecated: {name}@{version}')
                    continue
                with TemporaryDirectory() as tempdir:
                    info = self._extract_package(f'{name}@{version}', tempdir)
                if _validate_extension(info['data']):
                    return
                return version

    def latest_compatible_package_versions(self, names):
        """Get the latest compatible versions of several packages

        Like _latest_compatible_package_version, but optimized for
        retrieving the latest version for several packages in one go.
        """
        core_data = self.info['core_data']
        keys = []
        for name in names:
            try:
                metadata = _fetch_package_metadata(self.registry, name, self.logger)
            except URLError:
                continue
            versions = metadata.get('versions', {})

            def sort_key(key_value):
                return _semver_key(key_value[0], prerelease_first=True)
            for version, data in sorted(versions.items(), key=sort_key, reverse=True):
                if 'deprecated' in data:
                    continue
                deps = data.get('dependencies', {})
                errors = _validate_compatibility(name, deps, core_data)
                if not errors:
                    keys.append(f'{name}@{version}')
                    break
        versions = {}
        if not keys:
            return versions
        with TemporaryDirectory() as tempdir:
            ret = self._run([which('npm'), 'pack', *keys], cwd=tempdir)
            if ret != 0:
                msg = '"%s" is not a valid npm package'
                raise ValueError(msg % keys)
            for key in keys:
                fname = key[0].replace('@', '') + key[1:].replace('@', '-').replace('/', '-') + '.tgz'
                data = read_package(osp.join(tempdir, fname))
                if not _validate_extension(data):
                    versions[data['name']] = data['version']
        return versions

    def _format_no_compatible_package_version(self, name):
        """Get the latest compatible version of a package"""
        core_data = self.info['core_data']
        lab_newer_than_latest = False
        latest_newer_than_lab = False
        try:
            metadata = _fetch_package_metadata(self.registry, name, self.logger)
        except URLError:
            pass
        else:
            versions = metadata.get('versions', {})

            def sort_key(key_value):
                return _semver_key(key_value[0], prerelease_first=True)
            store = tuple(sorted(versions.items(), key=sort_key, reverse=True))
            latest_deps = store[0][1].get('dependencies', {})
            core_deps = core_data['resolutions']
            singletons = core_data['jupyterlab']['singletonPackages']
            for key, value in latest_deps.items():
                if key in singletons:
                    c = _compare_ranges(core_deps[key], value, drop_prerelease1=True)
                    lab_newer_than_latest = lab_newer_than_latest or c < 0
                    latest_newer_than_lab = latest_newer_than_lab or c > 0
        if lab_newer_than_latest:
            return 'The extension "%s" does not yet support the current version of JupyterLab.\n' % name
        parts = ['No version of {extension} could be found that is compatible with the current version of JupyterLab.']
        if latest_newer_than_lab:
            parts.extend(('However, it seems to support a new version of JupyterLab.', 'Consider upgrading JupyterLab.'))
        return ' '.join(parts).format(extension=name)

    def _run(self, cmd, **kwargs):
        """Run the command using our logger and abort callback.

        Returns the exit code.
        """
        if self.kill_event.is_set():
            msg = 'Command was killed'
            raise ValueError(msg)
        kwargs['logger'] = self.logger
        kwargs['kill_event'] = self.kill_event
        proc = ProgressProcess(cmd, **kwargs)
        return proc.wait()