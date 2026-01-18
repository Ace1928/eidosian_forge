import weakref
from time import perf_counter
from .functions import SignalBlock
from .Qt import QtCore
from .ThreadsafeTimer import ThreadsafeTimer
Return a SignalBlocker that temporarily blocks input signals to
        this proxy.
        