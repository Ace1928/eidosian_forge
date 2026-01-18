import contextlib
from numba.core.utils import threadsafe_cached_property as cached_property
from numba.core.descriptors import TargetDescriptor
from numba.core import utils, typing, dispatcher, cpu
@cached_property
def _toplevel_typing_context(self):
    return typing.Context()