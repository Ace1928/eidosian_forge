from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePath
import os
import typing as T
from . import NewExtensionModule, ModuleInfo
from . import ModuleReturnValue
from .. import build
from .. import dependencies
from .. import mesonlib
from .. import mlog
from ..coredata import BUILTIN_DIR_OPTIONS
from ..dependencies.pkgconfig import PkgConfigDependency, PkgConfigInterface
from ..interpreter.type_checking import D_MODULE_VERSIONS_KW, INSTALL_DIR_KW, VARIABLES_KW, NoneType
from ..interpreterbase import FeatureNew, FeatureDeprecated
from ..interpreterbase.decorators import ContainerTypeInfo, KwargInfo, typed_kwargs, typed_pos_args
def _generate_pkgconfig_file(self, state: ModuleState, deps: DependenciesHelper, subdirs: T.List[str], name: str, description: str, url: str, version: str, pcfile: str, conflicts: T.List[str], variables: T.List[T.Tuple[str, str]], unescaped_variables: T.List[T.Tuple[str, str]], uninstalled: bool=False, dataonly: bool=False, pkgroot: T.Optional[str]=None) -> None:
    coredata = state.environment.get_coredata()
    referenced_vars = set()
    optnames = [x.name for x in BUILTIN_DIR_OPTIONS.keys()]
    if not dataonly:
        referenced_vars |= {'prefix', 'includedir'}
        if deps.pub_libs or deps.priv_libs:
            referenced_vars |= {'libdir'}
    implicit_vars_warning = False
    redundant_vars_warning = False
    varnames = set()
    varstrings = set()
    for k, v in variables + unescaped_variables:
        varnames |= {k}
        varstrings |= {v}
    for optname in optnames:
        optvar = f'${{{optname}}}'
        if any((x.startswith(optvar) for x in varstrings)):
            if optname in varnames:
                redundant_vars_warning = True
            else:
                if dataonly or optname not in {'prefix', 'includedir', 'libdir'}:
                    implicit_vars_warning = True
                referenced_vars |= {'prefix', optname}
    if redundant_vars_warning:
        FeatureDeprecated.single_use('pkgconfig.generate variable for builtin directories', '0.62.0', state.subproject, 'They will be automatically included when referenced', state.current_node)
    if implicit_vars_warning:
        FeatureNew.single_use('pkgconfig.generate implicit variable for builtin directories', '0.62.0', state.subproject, location=state.current_node)
    if uninstalled:
        outdir = os.path.join(state.environment.build_dir, 'meson-uninstalled')
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        prefix = PurePath(state.environment.get_build_dir())
        srcdir = PurePath(state.environment.get_source_dir())
    else:
        outdir = state.environment.scratch_dir
        prefix = PurePath(_as_str(coredata.get_option(mesonlib.OptionKey('prefix'))))
        if pkgroot:
            pkgroot_ = PurePath(pkgroot)
            if not pkgroot_.is_absolute():
                pkgroot_ = prefix / pkgroot
            elif prefix not in pkgroot_.parents:
                raise mesonlib.MesonException(f'Pkgconfig prefix cannot be outside of the prefix when pkgconfig.relocatable=true. Pkgconfig prefix is {pkgroot_.as_posix()}.')
            prefix = PurePath('${pcfiledir}', os.path.relpath(prefix, pkgroot_))
    fname = os.path.join(outdir, pcfile)
    with open(fname, 'w', encoding='utf-8') as ofile:
        for optname in optnames:
            if optname in referenced_vars - varnames:
                if optname == 'prefix':
                    ofile.write('prefix={}\n'.format(self._escape(prefix)))
                else:
                    dirpath = PurePath(_as_str(coredata.get_option(mesonlib.OptionKey(optname))))
                    ofile.write('{}={}\n'.format(optname, self._escape('${prefix}' / dirpath)))
        if uninstalled and (not dataonly):
            ofile.write('srcdir={}\n'.format(self._escape(srcdir)))
        if variables or unescaped_variables:
            ofile.write('\n')
        for k, v in variables:
            ofile.write('{}={}\n'.format(k, self._escape(v)))
        for k, v in unescaped_variables:
            ofile.write(f'{k}={v}\n')
        ofile.write('\n')
        ofile.write(f'Name: {name}\n')
        if len(description) > 0:
            ofile.write(f'Description: {description}\n')
        if len(url) > 0:
            ofile.write(f'URL: {url}\n')
        ofile.write(f'Version: {version}\n')
        reqs_str = deps.format_reqs(deps.pub_reqs)
        if len(reqs_str) > 0:
            ofile.write(f'Requires: {reqs_str}\n')
        reqs_str = deps.format_reqs(deps.priv_reqs)
        if len(reqs_str) > 0:
            ofile.write(f'Requires.private: {reqs_str}\n')
        if len(conflicts) > 0:
            ofile.write('Conflicts: {}\n'.format(' '.join(conflicts)))

        def generate_libs_flags(libs: T.List[LIBS]) -> T.Iterable[str]:
            msg = "Library target {0!r} has {1!r} set. Compilers may not find it from its '-l{2}' linker flag in the {3!r} pkg-config file."
            Lflags = []
            for l in libs:
                if isinstance(l, str):
                    yield l
                else:
                    install_dir: T.Union[str, bool]
                    if uninstalled:
                        install_dir = os.path.dirname(state.backend.get_target_filename_abs(l))
                    else:
                        _i = l.get_custom_install_dir()
                        install_dir = _i[0] if _i else None
                    if install_dir is False:
                        continue
                    if isinstance(l, build.BuildTarget) and 'cs' in l.compilers:
                        if isinstance(install_dir, str):
                            Lflag = '-r{}/{}'.format(self._escape(self._make_relative(prefix, install_dir)), l.filename)
                        else:
                            Lflag = '-r${libdir}/%s' % l.filename
                    elif isinstance(install_dir, str):
                        Lflag = '-L{}'.format(self._escape(self._make_relative(prefix, install_dir)))
                    else:
                        Lflag = '-L${libdir}'
                    if Lflag not in Lflags:
                        Lflags.append(Lflag)
                        yield Lflag
                    lname = self._get_lname(l, msg, pcfile)
                    if isinstance(l, build.BuildTarget) and l.name_suffix_set:
                        mlog.warning(msg.format(l.name, 'name_suffix', lname, pcfile))
                    if isinstance(l, (build.CustomTarget, build.CustomTargetIndex)) or 'cs' not in l.compilers:
                        yield f'-l{lname}'
        if len(deps.pub_libs) > 0:
            ofile.write('Libs: {}\n'.format(' '.join(generate_libs_flags(deps.pub_libs))))
        if len(deps.priv_libs) > 0:
            ofile.write('Libs.private: {}\n'.format(' '.join(generate_libs_flags(deps.priv_libs))))
        cflags: T.List[str] = []
        if uninstalled:
            for d in deps.uninstalled_incdirs:
                for basedir in ['${prefix}', '${srcdir}']:
                    path = self._escape(PurePath(basedir, d).as_posix())
                    cflags.append(f'-I{path}')
        else:
            for d in subdirs:
                if d == '.':
                    cflags.append('-I${includedir}')
                else:
                    cflags.append(self._escape(PurePath('-I${includedir}') / d))
        cflags += [self._escape(f) for f in deps.cflags]
        if cflags and (not dataonly):
            ofile.write('Cflags: {}\n'.format(' '.join(cflags)))