from collections import namedtuple
import copy
import warnings
from numba.core.tracing import event
from numba.core import (utils, errors, interpreter, bytecode, postproc, config,
from numba.parfors.parfor import ParforDiagnostics
from numba.core.errors import CompilerError
from numba.core.environment import lookup_environment
from numba.core.compiler_machinery import PassManager
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
from numba.core.object_mode_passes import (ObjectModeFrontEnd,
from numba.core.targetconfig import TargetConfig, Option, ConfigStack
class CompileResult(namedtuple('_CompileResult', CR_FIELDS)):
    """
    A structure holding results from the compilation of a function.
    """
    __slots__ = ()

    def _reduce(self):
        """
        Reduce a CompileResult to picklable components.
        """
        libdata = self.library.serialize_using_object_code()
        typeann = str(self.type_annotation)
        fndesc = self.fndesc
        fndesc.typemap = fndesc.calltypes = None
        referenced_envs = self._find_referenced_environments()
        return (libdata, self.fndesc, self.environment, self.signature, self.objectmode, self.lifted, typeann, self.reload_init, tuple(referenced_envs))

    def _find_referenced_environments(self):
        """Returns a list of referenced environments
        """
        mod = self.library._final_module
        referenced_envs = []
        for gv in mod.global_variables:
            gvn = gv.name
            if gvn.startswith('_ZN08NumbaEnv'):
                env = lookup_environment(gvn)
                if env is not None:
                    if env.can_cache():
                        referenced_envs.append(env)
        return referenced_envs

    @classmethod
    def _rebuild(cls, target_context, libdata, fndesc, env, signature, objectmode, lifted, typeann, reload_init, referenced_envs):
        if reload_init:
            for fn in reload_init:
                fn()
        library = target_context.codegen().unserialize_library(libdata)
        cfunc = target_context.get_executable(library, fndesc, env)
        cr = cls(target_context=target_context, typing_context=target_context.typing_context, library=library, environment=env, entry_point=cfunc, fndesc=fndesc, type_annotation=typeann, signature=signature, objectmode=objectmode, lifted=lifted, typing_error=None, call_helper=None, metadata=None, reload_init=reload_init, referenced_envs=referenced_envs)
        for env in referenced_envs:
            library.codegen.set_env(env.env_name, env)
        return cr

    @property
    def codegen(self):
        return self.target_context.codegen()

    def dump(self, tab=''):
        print(f'{tab}DUMP {type(self).__name__} {self.entry_point}')
        self.signature.dump(tab=tab + '  ')
        print(f'{tab}END DUMP')