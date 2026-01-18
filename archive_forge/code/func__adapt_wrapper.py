import inspect
from . import ctraits
from .constants import ComparisonMode, DefaultValue, default_value_map
from .observation.i_observable import IObservable
from .trait_base import SequenceTypes, Undefined
from .trait_dict_object import TraitDictObject
from .trait_list_object import TraitListObject
from .trait_set_object import TraitSetObject
def _adapt_wrapper(*args, **kw):
    from traits.adaptation.api import adapt
    ctraits._adapt(adapt)
    return adapt(*args, **kw)