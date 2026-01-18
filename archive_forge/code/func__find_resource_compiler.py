from __future__ import annotations
import enum
import os
import re
import typing as T
from . import ExtensionModule, ModuleInfo
from . import ModuleReturnValue
from .. import mesonlib, build
from .. import mlog
from ..interpreter.type_checking import DEPEND_FILES_KW, DEPENDS_KW, INCLUDE_DIRECTORIES
from ..interpreterbase.decorators import ContainerTypeInfo, FeatureNew, KwargInfo, typed_kwargs, typed_pos_args
from ..mesonlib import MachineChoice, MesonException
from ..programs import ExternalProgram
def _find_resource_compiler(self, state: 'ModuleState') -> T.Tuple[ExternalProgram, ResourceCompilerType]:
    for_machine = MachineChoice.HOST
    if self._rescomp:
        return self._rescomp
    rescomp = ExternalProgram.from_bin_list(state.environment, for_machine, 'windres')
    if not rescomp or not rescomp.found():
        comp = self.detect_compiler(state.environment.coredata.compilers[for_machine])
        if comp.id in {'msvc', 'clang-cl', 'intel-cl'} or (comp.linker and comp.linker.id in {'link', 'lld-link'}):
            rescomp = ExternalProgram('rc', silent=True)
        else:
            rescomp = ExternalProgram('windres', silent=True)
    if not rescomp.found():
        raise MesonException('Could not find Windows resource compiler')
    for arg, match, rc_type in [('/?', '^.*Microsoft.*Resource Compiler.*$', ResourceCompilerType.rc), ('/?', 'LLVM Resource Converter.*$', ResourceCompilerType.rc), ('--version', '^.*GNU windres.*$', ResourceCompilerType.windres), ('--version', '^.*Wine Resource Compiler.*$', ResourceCompilerType.wrc)]:
        p, o, e = mesonlib.Popen_safe(rescomp.get_command() + [arg])
        m = re.search(match, o, re.MULTILINE)
        if m:
            mlog.log('Windows resource compiler: %s' % m.group())
            self._rescomp = (rescomp, rc_type)
            break
    else:
        raise MesonException('Could not determine type of Windows resource compiler')
    return self._rescomp