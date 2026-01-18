from importlib import import_module
import sys
from types import FunctionType, MethodType
from .constants import DefaultValue, ValidateTrait
from .trait_base import (
from .trait_base import RangeTypes  # noqa: F401, used by TraitsUI
from .trait_errors import TraitError
from .trait_dict_object import TraitDictEvent, TraitDictObject
from .trait_converters import trait_from
from .trait_handler import TraitHandler
from .trait_list_object import TraitListEvent, TraitListObject
from .util.deprecated import deprecated
class TraitTuple(TraitHandler):
    """ Ensures that values assigned to a trait attribute are tuples of a
    specified length, with elements that are of specified types.

    TraitTuple is the underlying handler for the predefined trait **Tuple**,
    and the trait factory Tuple().

    Example
    -------

    The following example defines a ``Card`` class::

        rank = Range(1, 13)
        suit = Trait('Hearts', 'Diamonds', 'Spades', 'Clubs')
        class Card(HasTraits):
            value = Trait(TraitTuple(rank, suit))

    The Card class has a **value** trait attribute,
    which must be a tuple of two elments. The first element must be an integer
    in the range from 1 to 13, and the second element must be one of the four
    strings, 'Hearts', 'Diamonds', 'Spades', or 'Clubs'.

    Parameters
    ----------
    *args
        The traits, each *trait*\\ :sub:`i` specifies the type that
        the *i*\\ th element of a tuple must be.  Each *trait*\\ :sub:`i`
        must be either a trait, or a value that can be
        converted to a trait using the trait_from() function. The resulting
        trait handler accepts values that are tuples of the same length as
        *args*, and whose *i*\\ th element is of the type specified by
        *trait*\\ :sub:`i`.

    Parameters
    ----------
    types : tuple of CTrait instances
        The traits to use for each item in a validated tuple.
    """

    @deprecated(_WARNING_FORMAT_STR.format(handler='TraitTuple', replacement='Tuple'))
    def __init__(self, *args):
        self.types = tuple([trait_from(arg) for arg in args])
        self.fast_validate = (ValidateTrait.tuple, self.types)

    def validate(self, object, name, value):
        try:
            if isinstance(value, tuple):
                types = self.types
                if len(value) == len(types):
                    values = []
                    for i, type in enumerate(types):
                        values.append(type.handler.validate(object, name, value[i]))
                    return tuple(values)
        except:
            pass
        self.error(object, name, value)

    def full_info(self, object, name, value):
        return 'a tuple of the form: (%s)' % ', '.join([self._trait_info(type, object, name, value) for type in self.types])

    def _trait_info(self, type, object, name, value):
        handler = type.handler
        if handler is None:
            return 'any value'
        return handler.full_info(object, name, value)

    def get_editor(self, trait):
        from traitsui.api import TupleEditor
        return TupleEditor(types=self.types, labels=trait.labels or [], cols=trait.cols or 1)