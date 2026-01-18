from abc import abstractmethod
from abc import ABCMeta
import threading
import time
import uuid
def _maybe_copy(source, target, attr):
    value = getattr(source, attr, source)
    if value is not source:
        setattr(target, attr, value)