import contextlib
from numba.core.utils import threadsafe_cached_property as cached_property
from numba.core.descriptors import TargetDescriptor
from numba.core import utils, typing, dispatcher, cpu
class CPUDispatcher(dispatcher.Dispatcher):
    targetdescr = cpu_target