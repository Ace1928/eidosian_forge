from __future__ import annotations
from .base import ExternalDependency, DependencyException, DependencyTypeName
from .pkgconfig import PkgConfigDependency
from ..mesonlib import (Popen_safe, OptionKey, join_args, version_compare)
from ..programs import ExternalProgram
from .. import mlog
import re
import os
import json
import typing as T
class DubDependency(ExternalDependency):
    class_dubbin: T.Optional[T.Tuple[ExternalProgram, str]] = None
    class_dubbin_searched = False

    def __init__(self, name: str, environment: 'Environment', kwargs: T.Dict[str, T.Any]):
        super().__init__(DependencyTypeName('dub'), environment, kwargs, language='d')
        self.name = name
        from ..compilers.d import DCompiler, d_feature_args
        _temp_comp = super().get_compiler()
        assert isinstance(_temp_comp, DCompiler)
        self.compiler = _temp_comp
        if 'required' in kwargs:
            self.required = kwargs.get('required')
        if DubDependency.class_dubbin is None and (not DubDependency.class_dubbin_searched):
            DubDependency.class_dubbin = self._check_dub()
            DubDependency.class_dubbin_searched = True
        if DubDependency.class_dubbin is None:
            if self.required:
                raise DependencyException('DUB not found.')
            self.is_found = False
            return
        self.dubbin, dubver = DubDependency.class_dubbin
        assert isinstance(self.dubbin, ExternalProgram)
        if version_compare(dubver, '>1.31.1'):
            if self.required:
                raise DependencyException(f"DUB version {dubver} is not compatible with Meson (can't locate artifacts in Dub cache)")
            self.is_found = False
            return
        mlog.debug('Determining dependency {!r} with DUB executable {!r}'.format(name, self.dubbin.get_path()))
        main_pack_spec = name
        if 'version' in kwargs:
            version_spec = kwargs['version']
            if isinstance(version_spec, list):
                version_spec = ' '.join(version_spec)
            main_pack_spec = f'{name}@{version_spec}'
        dub_arch = self.compiler.arch
        dub_buildtype = str(environment.coredata.get_option(OptionKey('buildtype')))
        if dub_buildtype == 'debugoptimized':
            dub_buildtype = 'release-debug'
        elif dub_buildtype == 'minsize':
            dub_buildtype = 'release'
        describe_cmd = ['describe', main_pack_spec, '--arch=' + dub_arch, '--build=' + dub_buildtype, '--compiler=' + self.compiler.get_exelist()[-1]]
        ret, res, err = self._call_dubbin(describe_cmd)
        if ret != 0:
            mlog.debug('DUB describe failed: ' + err)
            if 'locally' in err:
                fetch_cmd = ['dub', 'fetch', main_pack_spec]
                mlog.error(mlog.bold(main_pack_spec), 'is not present locally. You may try the following command:')
                mlog.log(mlog.bold(join_args(fetch_cmd)))
            self.is_found = False
            return

        def dub_build_deep_command() -> str:
            cmd = ['dub', 'run', 'dub-build-deep', '--yes', '--', main_pack_spec, '--arch=' + dub_arch, '--compiler=' + self.compiler.get_exelist()[-1], '--build=' + dub_buildtype]
            return join_args(cmd)
        dub_comp_id = self.compiler.get_id().replace('llvm', 'ldc').replace('gcc', 'gdc')
        description = json.loads(res)
        self.compile_args = []
        self.link_args = self.raw_link_args = []
        show_buildtype_warning = False

        def find_package_target(pkg: T.Dict[str, str]) -> bool:
            nonlocal show_buildtype_warning
            pack_id = f'{pkg['name']}@{pkg['version']}'
            tgt_file, compatibilities = self._find_compatible_package_target(description, pkg, dub_comp_id)
            if tgt_file is None:
                if not compatibilities:
                    mlog.error(mlog.bold(pack_id), 'not found')
                elif 'compiler' not in compatibilities:
                    mlog.error(mlog.bold(pack_id), 'found but not compiled with ', mlog.bold(dub_comp_id))
                elif dub_comp_id != 'gdc' and 'compiler_version' not in compatibilities:
                    mlog.error(mlog.bold(pack_id), 'found but not compiled with', mlog.bold(f'{dub_comp_id}-{self.compiler.version}'))
                elif 'arch' not in compatibilities:
                    mlog.error(mlog.bold(pack_id), 'found but not compiled for', mlog.bold(dub_arch))
                elif 'platform' not in compatibilities:
                    mlog.error(mlog.bold(pack_id), 'found but not compiled for', mlog.bold(description['platform'].join('.')))
                elif 'configuration' not in compatibilities:
                    mlog.error(mlog.bold(pack_id), 'found but not compiled for the', mlog.bold(pkg['configuration']), 'configuration')
                else:
                    mlog.error(mlog.bold(pack_id), 'not found')
                mlog.log('You may try the following command to install the necessary DUB libraries:')
                mlog.log(mlog.bold(dub_build_deep_command()))
                return False
            if 'build_type' not in compatibilities:
                mlog.warning(mlog.bold(pack_id), 'found but not compiled as', mlog.bold(dub_buildtype))
                show_buildtype_warning = True
            self.link_args.append(tgt_file)
            return True
        self.is_found = False
        packages = {}
        for pkg in description['packages']:
            packages[pkg['name']] = pkg
            if not pkg['active']:
                continue
            if pkg['targetType'] == 'dynamicLibrary':
                mlog.error('DUB dynamic library dependencies are not supported.')
                self.is_found = False
                return
            if pkg['name'] == name:
                self.is_found = True
                if pkg['targetType'] not in ['library', 'sourceLibrary', 'staticLibrary']:
                    mlog.error(mlog.bold(name), "found but it isn't a library")
                    self.is_found = False
                    return
                self.version = pkg['version']
                self.pkg = pkg
        targets = {}
        for tgt in description['targets']:
            targets[tgt['rootPackage']] = tgt
        if name not in targets:
            self.is_found = False
            if self.pkg['targetType'] == 'sourceLibrary':
                mlog.error('DUB targets of type', mlog.bold('sourceLibrary'), 'are not supported.')
            else:
                mlog.error('Could not find target description for', mlog.bold(main_pack_spec))
        if not self.is_found:
            mlog.error(f'Could not find {name} in DUB description')
            return
        self.static = True
        if not find_package_target(self.pkg):
            self.is_found = False
            return
        for link_dep in targets[name]['linkDependencies']:
            pkg = packages[link_dep]
            if not find_package_target(pkg):
                self.is_found = False
                return
        if show_buildtype_warning:
            mlog.log('If it is not suitable, try the following command and reconfigure Meson with', mlog.bold('--clearcache'))
            mlog.log(mlog.bold(dub_build_deep_command()))
        bs = targets[name]['buildSettings']
        for flag in bs['dflags']:
            self.compile_args.append(flag)
        for path in bs['importPaths']:
            self.compile_args.append('-I' + path)
        for path in bs['stringImportPaths']:
            if 'import_dir' not in d_feature_args[self.compiler.id]:
                break
            flag = d_feature_args[self.compiler.id]['import_dir']
            self.compile_args.append(f'{flag}={path}')
        for ver in bs['versions']:
            if 'version' not in d_feature_args[self.compiler.id]:
                break
            flag = d_feature_args[self.compiler.id]['version']
            self.compile_args.append(f'{flag}={ver}')
        if bs['mainSourceFile']:
            self.compile_args.append(bs['mainSourceFile'])
        for file in bs['sourceFiles']:
            if file.endswith('.lib') or file.endswith('.a'):
                self.link_args.append(file)
        for flag in bs['lflags']:
            self.link_args.append(flag)
        is_windows = self.env.machines.host.is_windows()
        if is_windows:
            winlibs = ['kernel32', 'user32', 'gdi32', 'winspool', 'shell32', 'ole32', 'oleaut32', 'uuid', 'comdlg32', 'advapi32', 'ws2_32']
        for lib in bs['libs']:
            if os.name != 'nt':
                pkgdep = PkgConfigDependency(lib, environment, {'required': 'true', 'silent': 'true'})
                if pkgdep.is_found:
                    for arg in pkgdep.get_compile_args():
                        self.compile_args.append(arg)
                    for arg in pkgdep.get_link_args():
                        self.link_args.append(arg)
                    for arg in pkgdep.get_link_args(raw=True):
                        self.raw_link_args.append(arg)
                    continue
            if is_windows and lib in winlibs:
                self.link_args.append(lib + '.lib')
                continue
            self.link_args.append('-l' + lib)

    def _find_compatible_package_target(self, jdesc: T.Dict[str, str], jpack: T.Dict[str, str], dub_comp_id: str) -> T.Tuple[str, T.Set[str]]:
        dub_build_path = os.path.join(jpack['path'], '.dub', 'build')
        if not os.path.exists(dub_build_path):
            return (None, None)
        conf = jpack['configuration']
        build_type = jdesc['buildType']
        platforms = jdesc['platform']
        archs = jdesc['architecture']
        comp_versions = []
        if dub_comp_id != 'gdc':
            comp_versions.append(self.compiler.version)
            ret, res = self._call_compbin(['--version'])[0:2]
            if ret != 0:
                mlog.error('Failed to run {!r}', mlog.bold(dub_comp_id))
                return (None, None)
            d_ver_reg = re.search('v[0-9].[0-9][0-9][0-9].[0-9]', res)
            if d_ver_reg is not None:
                frontend_version = d_ver_reg.group()
                frontend_id = frontend_version.rsplit('.', 1)[0].replace('v', '').replace('.', '')
                comp_versions.extend([frontend_version, frontend_id])
        compatibilities: T.Set[str] = set()
        check_list = ('configuration', 'platform', 'arch', 'compiler', 'compiler_version')
        for entry in os.listdir(dub_build_path):
            target = os.path.join(dub_build_path, entry, jpack['targetFileName'])
            if not os.path.exists(target):
                mlog.debug('WARNING: Could not find a Dub target: ' + target)
                continue
            comps = set()
            if conf in entry:
                comps.add('configuration')
            if build_type in entry:
                comps.add('build_type')
            if all((platform in entry for platform in platforms)):
                comps.add('platform')
            if all((arch in entry for arch in archs)):
                comps.add('arch')
            if dub_comp_id in entry:
                comps.add('compiler')
            if dub_comp_id == 'gdc' or any((cv in entry for cv in comp_versions)):
                comps.add('compiler_version')
            if all((key in comps for key in check_list)):
                return (target, comps)
            else:
                compatibilities = set.union(compatibilities, comps)
        return (None, compatibilities)

    def _call_dubbin(self, args: T.List[str], env: T.Optional[T.Dict[str, str]]=None) -> T.Tuple[int, str, str]:
        assert isinstance(self.dubbin, ExternalProgram)
        p, out, err = Popen_safe(self.dubbin.get_command() + args, env=env)
        return (p.returncode, out.strip(), err.strip())

    def _call_compbin(self, args: T.List[str], env: T.Optional[T.Dict[str, str]]=None) -> T.Tuple[int, str, str]:
        p, out, err = Popen_safe(self.compiler.get_exelist() + args, env=env)
        return (p.returncode, out.strip(), err.strip())

    def _check_dub(self) -> T.Optional[T.Tuple[ExternalProgram, str]]:

        def find() -> T.Optional[T.Tuple[ExternalProgram, str]]:
            dubbin = ExternalProgram('dub', silent=True)
            if not dubbin.found():
                return None
            try:
                p, out = Popen_safe(dubbin.get_command() + ['--version'])[0:2]
                if p.returncode != 0:
                    mlog.warning("Found dub {!r} but couldn't run it".format(' '.join(dubbin.get_command())))
                    return None
            except (FileNotFoundError, PermissionError):
                return None
            vermatch = re.search('DUB version (\\d+\\.\\d+\\.\\d+.*), ', out.strip())
            if vermatch:
                dubver = vermatch.group(1)
            else:
                mlog.warning(f"Found dub {' '.join(dubbin.get_command())} but couldn't parse version in {out.strip()}")
                return None
            return (dubbin, dubver)
        found = find()
        if found is None:
            mlog.log('Found DUB:', mlog.red('NO'))
        else:
            dubbin, dubver = found
            mlog.log('Found DUB:', mlog.bold(dubbin.get_path()), '(version %s)' % dubver)
        return found