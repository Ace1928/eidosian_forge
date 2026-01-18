from __future__ import annotations
import copy
import os
import collections
import itertools
import typing as T
from enum import Enum
from .. import mlog, mesonlib
from ..compilers import clib_langs
from ..mesonlib import LibType, MachineChoice, MesonException, HoldableObject, OptionKey
from ..mesonlib import version_compare_many
class InternalDependency(Dependency):

    def __init__(self, version: str, incdirs: T.List['IncludeDirs'], compile_args: T.List[str], link_args: T.List[str], libraries: T.List[LibTypes], whole_libraries: T.List[T.Union[StaticLibrary, CustomTarget, CustomTargetIndex]], sources: T.Sequence[T.Union[mesonlib.File, GeneratedTypes, StructuredSources]], extra_files: T.Sequence[mesonlib.File], ext_deps: T.List[Dependency], variables: T.Dict[str, str], d_module_versions: T.List[T.Union[str, int]], d_import_dirs: T.List['IncludeDirs'], objects: T.List['ExtractedObjects']):
        super().__init__(DependencyTypeName('internal'), {})
        self.version = version
        self.is_found = True
        self.include_directories = incdirs
        self.compile_args = compile_args
        self.link_args = link_args
        self.libraries = libraries
        self.whole_libraries = whole_libraries
        self.sources = list(sources)
        self.extra_files = list(extra_files)
        self.ext_deps = ext_deps
        self.variables = variables
        self.objects = objects
        if d_module_versions:
            self.d_features['versions'] = d_module_versions
        if d_import_dirs:
            self.d_features['import_dirs'] = d_import_dirs

    def __deepcopy__(self, memo: T.Dict[int, 'InternalDependency']) -> 'InternalDependency':
        result = self.__class__.__new__(self.__class__)
        assert isinstance(result, InternalDependency)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in {'libraries', 'whole_libraries'}:
                setattr(result, k, copy.copy(v))
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def summary_value(self) -> mlog.AnsiDecorator:
        return mlog.green('YES')

    def is_built(self) -> bool:
        if self.sources or self.libraries or self.whole_libraries:
            return True
        return any((d.is_built() for d in self.ext_deps))

    def get_partial_dependency(self, *, compile_args: bool=False, link_args: bool=False, links: bool=False, includes: bool=False, sources: bool=False, extra_files: bool=False) -> InternalDependency:
        final_compile_args = self.compile_args.copy() if compile_args else []
        final_link_args = self.link_args.copy() if link_args else []
        final_libraries = self.libraries.copy() if links else []
        final_whole_libraries = self.whole_libraries.copy() if links else []
        final_sources = self.sources.copy() if sources else []
        final_extra_files = self.extra_files.copy() if extra_files else []
        final_includes = self.include_directories.copy() if includes else []
        final_deps = [d.get_partial_dependency(compile_args=compile_args, link_args=link_args, links=links, includes=includes, sources=sources) for d in self.ext_deps]
        return InternalDependency(self.version, final_includes, final_compile_args, final_link_args, final_libraries, final_whole_libraries, final_sources, final_extra_files, final_deps, self.variables, [], [], [])

    def get_include_dirs(self) -> T.List['IncludeDirs']:
        return self.include_directories

    def get_variable(self, *, cmake: T.Optional[str]=None, pkgconfig: T.Optional[str]=None, configtool: T.Optional[str]=None, internal: T.Optional[str]=None, default_value: T.Optional[str]=None, pkgconfig_define: PkgConfigDefineType=None) -> str:
        val = self.variables.get(internal, default_value)
        if val is not None:
            return val
        raise DependencyException(f'Could not get an internal variable and no default provided for {self!r}')

    def generate_link_whole_dependency(self) -> Dependency:
        from ..build import SharedLibrary, CustomTarget, CustomTargetIndex
        new_dep = copy.deepcopy(self)
        for x in new_dep.libraries:
            if isinstance(x, SharedLibrary):
                raise MesonException('Cannot convert a dependency to link_whole when it contains a SharedLibrary')
            elif isinstance(x, (CustomTarget, CustomTargetIndex)) and x.links_dynamically():
                raise MesonException('Cannot convert a dependency to link_whole when it contains a CustomTarget or CustomTargetIndex which is a shared library')
        new_dep.whole_libraries += T.cast('T.List[T.Union[StaticLibrary, CustomTarget, CustomTargetIndex]]', new_dep.libraries)
        new_dep.libraries = []
        return new_dep