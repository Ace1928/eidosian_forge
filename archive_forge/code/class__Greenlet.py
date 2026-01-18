from time import monotonic
from kombu.asynchronous import timer as _timer
from . import base
class _Greenlet(Greenlet):
    cancel = Greenlet.kill