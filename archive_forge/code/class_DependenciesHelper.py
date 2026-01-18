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
class DependenciesHelper:

    def __init__(self, state: ModuleState, name: str, metadata: T.Dict[str, MetaData]) -> None:
        self.state = state
        self.name = name
        self.metadata = metadata
        self.pub_libs: T.List[LIBS] = []
        self.pub_reqs: T.List[str] = []
        self.priv_libs: T.List[LIBS] = []
        self.priv_reqs: T.List[str] = []
        self.cflags: T.List[str] = []
        self.version_reqs: T.DefaultDict[str, T.Set[str]] = defaultdict(set)
        self.link_whole_targets: T.List[T.Union[build.CustomTarget, build.CustomTargetIndex, build.StaticLibrary]] = []
        self.uninstalled_incdirs: mesonlib.OrderedSet[str] = mesonlib.OrderedSet()

    def add_pub_libs(self, libs: T.List[ANY_DEP]) -> None:
        p_libs, reqs, cflags = self._process_libs(libs, True)
        self.pub_libs = p_libs + self.pub_libs
        self.pub_reqs += reqs
        self.cflags += cflags

    def add_priv_libs(self, libs: T.List[ANY_DEP]) -> None:
        p_libs, reqs, _ = self._process_libs(libs, False)
        self.priv_libs = p_libs + self.priv_libs
        self.priv_reqs += reqs

    def add_pub_reqs(self, reqs: T.List[T.Union[str, build.StaticLibrary, build.SharedLibrary, dependencies.Dependency]]) -> None:
        self.pub_reqs += self._process_reqs(reqs)

    def add_priv_reqs(self, reqs: T.List[T.Union[str, build.StaticLibrary, build.SharedLibrary, dependencies.Dependency]]) -> None:
        self.priv_reqs += self._process_reqs(reqs)

    def _check_generated_pc_deprecation(self, obj: T.Union[build.CustomTarget, build.CustomTargetIndex, build.StaticLibrary, build.SharedLibrary]) -> None:
        if obj.get_id() in self.metadata:
            return
        data = self.metadata[obj.get_id()]
        if data.warned:
            return
        mlog.deprecation('Library', mlog.bold(obj.name), 'was passed to the "libraries" keyword argument of a previous call to generate() method instead of first positional argument.', 'Adding', mlog.bold(data.display_name), 'to "Requires" field, but this is a deprecated behaviour that will change in a future version of Meson. Please report the issue if this warning cannot be avoided in your case.', location=data.location)
        data.warned = True

    def _process_reqs(self, reqs: T.Sequence[T.Union[str, build.StaticLibrary, build.SharedLibrary, dependencies.Dependency]]) -> T.List[str]:
        """Returns string names of requirements"""
        processed_reqs: T.List[str] = []
        for obj in mesonlib.listify(reqs):
            if not isinstance(obj, str):
                FeatureNew.single_use('pkgconfig.generate requirement from non-string object', '0.46.0', self.state.subproject)
            if isinstance(obj, (build.CustomTarget, build.CustomTargetIndex, build.SharedLibrary, build.StaticLibrary)) and obj.get_id() in self.metadata:
                self._check_generated_pc_deprecation(obj)
                processed_reqs.append(self.metadata[obj.get_id()].filebase)
            elif isinstance(obj, PkgConfigDependency):
                if obj.found():
                    processed_reqs.append(obj.name)
                    self.add_version_reqs(obj.name, obj.version_reqs)
            elif isinstance(obj, str):
                name, version_req = self.split_version_req(obj)
                processed_reqs.append(name)
                self.add_version_reqs(name, [version_req] if version_req is not None else None)
            elif isinstance(obj, dependencies.Dependency) and (not obj.found()):
                pass
            elif isinstance(obj, dependencies.ExternalDependency) and obj.name == 'threads':
                pass
            else:
                raise mesonlib.MesonException(f'requires argument not a string, library with pkgconfig-generated file or pkgconfig-dependency object, got {obj!r}')
        return processed_reqs

    def add_cflags(self, cflags: T.List[str]) -> None:
        self.cflags += mesonlib.stringlistify(cflags)

    def _add_uninstalled_incdirs(self, incdirs: T.List[build.IncludeDirs], subdir: T.Optional[str]=None) -> None:
        for i in incdirs:
            curdir = i.get_curdir()
            for d in i.get_incdirs():
                path = os.path.join(curdir, d)
                self.uninstalled_incdirs.add(path)
        if subdir is not None:
            self.uninstalled_incdirs.add(subdir)

    def _process_libs(self, libs: T.List[ANY_DEP], public: bool) -> T.Tuple[T.List[T.Union[str, build.SharedLibrary, build.StaticLibrary, build.CustomTarget, build.CustomTargetIndex]], T.List[str], T.List[str]]:
        libs = mesonlib.listify(libs)
        processed_libs: T.List[T.Union[str, build.SharedLibrary, build.StaticLibrary, build.CustomTarget, build.CustomTargetIndex]] = []
        processed_reqs: T.List[str] = []
        processed_cflags: T.List[str] = []
        for obj in libs:
            if isinstance(obj, (build.CustomTarget, build.CustomTargetIndex, build.SharedLibrary, build.StaticLibrary)) and obj.get_id() in self.metadata:
                self._check_generated_pc_deprecation(obj)
                processed_reqs.append(self.metadata[obj.get_id()].filebase)
            elif isinstance(obj, dependencies.ExternalDependency) and obj.name == 'valgrind':
                pass
            elif isinstance(obj, PkgConfigDependency):
                if obj.found():
                    processed_reqs.append(obj.name)
                    self.add_version_reqs(obj.name, obj.version_reqs)
            elif isinstance(obj, dependencies.InternalDependency):
                if obj.found():
                    if obj.objects:
                        raise mesonlib.MesonException('.pc file cannot refer to individual object files.')
                    processed_libs += obj.get_link_args()
                    processed_cflags += obj.get_compile_args()
                    self._add_lib_dependencies(obj.libraries, obj.whole_libraries, obj.ext_deps, public, private_external_deps=True)
                    self._add_uninstalled_incdirs(obj.get_include_dirs())
            elif isinstance(obj, dependencies.Dependency):
                if obj.found():
                    processed_libs += obj.get_link_args()
                    processed_cflags += obj.get_compile_args()
            elif isinstance(obj, build.SharedLibrary) and obj.shared_library_only:
                processed_libs.append(obj)
                self._add_uninstalled_incdirs(obj.get_include_dirs(), obj.get_subdir())
            elif isinstance(obj, (build.SharedLibrary, build.StaticLibrary)):
                processed_libs.append(obj)
                self._add_uninstalled_incdirs(obj.get_include_dirs(), obj.get_subdir())
                self._add_lib_dependencies(obj.link_targets, obj.link_whole_targets, obj.external_deps, isinstance(obj, build.StaticLibrary) and public)
            elif isinstance(obj, (build.CustomTarget, build.CustomTargetIndex)):
                if not obj.is_linkable_target():
                    raise mesonlib.MesonException('library argument contains a not linkable custom_target.')
                FeatureNew.single_use('custom_target in pkgconfig.generate libraries', '0.58.0', self.state.subproject)
                processed_libs.append(obj)
            elif isinstance(obj, str):
                processed_libs.append(obj)
            else:
                raise mesonlib.MesonException(f'library argument of type {type(obj).__name__} not a string, library or dependency object.')
        return (processed_libs, processed_reqs, processed_cflags)

    def _add_lib_dependencies(self, link_targets: T.Sequence[build.BuildTargetTypes], link_whole_targets: T.Sequence[T.Union[build.StaticLibrary, build.CustomTarget, build.CustomTargetIndex]], external_deps: T.List[dependencies.Dependency], public: bool, private_external_deps: bool=False) -> None:
        add_libs = self.add_pub_libs if public else self.add_priv_libs
        for t in link_targets:
            if t.is_internal():
                assert isinstance(t, (build.StaticLibrary, build.CustomTarget, build.CustomTargetIndex)), 'for mypy'
                self._add_link_whole(t, public)
            else:
                add_libs([t])
        for t in link_whole_targets:
            self._add_link_whole(t, public)
        if private_external_deps:
            self.add_priv_libs(T.cast('T.List[ANY_DEP]', external_deps))
        else:
            add_libs(T.cast('T.List[ANY_DEP]', external_deps))

    def _add_link_whole(self, t: T.Union[build.CustomTarget, build.CustomTargetIndex, build.StaticLibrary], public: bool) -> None:
        self.link_whole_targets.append(t)
        if isinstance(t, build.BuildTarget):
            self._add_lib_dependencies(t.link_targets, t.link_whole_targets, t.external_deps, public)

    def add_version_reqs(self, name: str, version_reqs: T.Optional[T.List[str]]) -> None:
        if version_reqs:
            self.version_reqs[name].update(version_reqs)

    def split_version_req(self, s: str) -> T.Tuple[str, T.Optional[str]]:
        for op in ['>=', '<=', '!=', '==', '=', '>', '<']:
            pos = s.find(op)
            if pos > 0:
                return (s[0:pos].strip(), s[pos:].strip())
        return (s, None)

    def format_vreq(self, vreq: str) -> str:
        for op in ['>=', '<=', '!=', '==', '=', '>', '<']:
            if vreq.startswith(op):
                return op + ' ' + vreq[len(op):]
        return vreq

    def format_reqs(self, reqs: T.List[str]) -> str:
        result: T.List[str] = []
        for name in reqs:
            vreqs = self.version_reqs.get(name, None)
            if vreqs:
                result += [name + ' ' + self.format_vreq(vreq) for vreq in sorted(vreqs)]
            else:
                result += [name]
        return ', '.join(result)

    def remove_dups(self) -> None:
        exclude: T.Set[str] = set()

        def _ids(x: T.Union[str, build.CustomTarget, build.CustomTargetIndex, build.StaticLibrary, build.SharedLibrary]) -> T.Iterable[str]:
            if isinstance(x, str):
                yield x
            else:
                if x.get_id() in self.metadata:
                    yield self.metadata[x.get_id()].display_name
                yield x.get_id()

        def _add_exclude(x: T.Union[str, build.CustomTarget, build.CustomTargetIndex, build.StaticLibrary, build.SharedLibrary]) -> bool:
            was_excluded = False
            for i in _ids(x):
                if i in exclude:
                    was_excluded = True
                else:
                    exclude.add(i)
            return was_excluded
        for t in self.link_whole_targets:
            _add_exclude(t)

        @T.overload
        def _fn(xs: T.List[str], libs: bool=False) -> T.List[str]:
            ...

        @T.overload
        def _fn(xs: T.List[LIBS], libs: bool=False) -> T.List[LIBS]:
            ...

        def _fn(xs: T.Union[T.List[str], T.List[LIBS]], libs: bool=False) -> T.Union[T.List[str], T.List[LIBS]]:
            result = []
            for x in xs:
                known_flags = ['-pthread']
                cannot_dedup = libs and isinstance(x, str) and (not x.startswith(('-l', '-L'))) and (x not in known_flags)
                if not cannot_dedup and _add_exclude(x):
                    continue
                result.append(x)
            return result
        self.pub_reqs = _fn(self.pub_reqs)
        self.pub_libs = _fn(self.pub_libs, True)
        self.priv_reqs = _fn(self.priv_reqs)
        self.priv_libs = _fn(self.priv_libs, True)
        exclude = set()
        self.cflags = _fn(self.cflags)