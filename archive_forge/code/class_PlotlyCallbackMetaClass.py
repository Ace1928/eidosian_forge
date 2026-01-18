from weakref import WeakValueDictionary
from ...element import Tiles
from ...streams import (
from .util import _trace_to_subplot
class PlotlyCallbackMetaClass(type):
    """
    Metaclass for PlotlyCallback classes.

    We want each callback class to keep track of all of the instances of the class.
    Using a meta class here lets us keep the logic for instance tracking in one place.
    """

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls.instances = WeakValueDictionary()

    def __call__(cls, *args, **kwargs):
        inst = super().__call__(*args, **kwargs)
        cls.instances[inst.plot.trace_uid] = inst
        return inst