from datetime import timedelta
import weakref
from collections import OrderedDict
from six.moves import _thread
class _TzFactory(type):

    def instance(cls, *args, **kwargs):
        """Alternate constructor that returns a fresh instance"""
        return type.__call__(cls, *args, **kwargs)