import contextlib
from numba.core.utils import threadsafe_cached_property as cached_property
from numba.core.descriptors import TargetDescriptor
from numba.core import utils, typing, dispatcher, cpu
class CPUTarget(TargetDescriptor):
    options = cpu.CPUTargetOptions

    @cached_property
    def _toplevel_target_context(self):
        return cpu.CPUContext(self.typing_context, self._target_name)

    @cached_property
    def _toplevel_typing_context(self):
        return typing.Context()

    @property
    def target_context(self):
        """
        The target context for CPU targets.
        """
        return self._toplevel_target_context

    @property
    def typing_context(self):
        """
        The typing context for CPU targets.
        """
        return self._toplevel_typing_context