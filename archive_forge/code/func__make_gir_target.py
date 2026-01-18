from __future__ import annotations
import copy
import itertools
import functools
import os
import subprocess
import textwrap
import typing as T
from . import (
from .. import build
from .. import interpreter
from .. import mesonlib
from .. import mlog
from ..build import CustomTarget, CustomTargetIndex, Executable, GeneratedList, InvalidArguments
from ..dependencies import Dependency, InternalDependency
from ..dependencies.pkgconfig import PkgConfigDependency, PkgConfigInterface
from ..interpreter.type_checking import DEPENDS_KW, DEPEND_FILES_KW, ENV_KW, INSTALL_DIR_KW, INSTALL_KW, NoneType, DEPENDENCY_SOURCES_KW, in_set_validator
from ..interpreterbase import noPosargs, noKwargs, FeatureNew, FeatureDeprecated
from ..interpreterbase import typed_kwargs, KwargInfo, ContainerTypeInfo
from ..interpreterbase.decorators import typed_pos_args
from ..mesonlib import (
from ..programs import OverrideProgram
from ..scripts.gettext import read_linguas
@staticmethod
def _make_gir_target(state: 'ModuleState', girfile: str, scan_command: T.Sequence[T.Union['FileOrString', Executable, ExternalProgram, OverrideProgram]], generated_files: T.Sequence[T.Union[str, mesonlib.File, CustomTarget, CustomTargetIndex, GeneratedList]], depends: T.Sequence[T.Union['FileOrString', build.BuildTarget, 'build.GeneratedTypes', build.StructuredSources]], kwargs: T.Dict[str, T.Any]) -> GirTarget:
    install = kwargs['install_gir']
    if install is None:
        install = kwargs['install']
    install_dir = kwargs['install_dir_gir']
    if install_dir is None:
        install_dir = os.path.join(state.environment.get_datadir(), 'gir-1.0')
    elif install_dir is False:
        install = False
    run_env = PkgConfigInterface.get_env(state.environment, MachineChoice.HOST, uninstalled=True)
    cc_exelist = state.environment.coredata.compilers.host['c'].get_exelist()
    run_env.set('CC', [quote_arg(x) for x in cc_exelist], ' ')
    run_env.merge(kwargs['env'])
    return GirTarget(girfile, state.subdir, state.subproject, state.environment, scan_command, generated_files, [girfile], build_by_default=kwargs['build_by_default'], extra_depends=depends, install=install, install_dir=[install_dir], install_tag=['devel'], env=run_env)