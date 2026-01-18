from collections import ChainMap
from itemloaders.common import wrap_loader_context
from itemloaders.utils import arg_to_iter
class TakeFirst:
    """
    Returns the first non-null/non-empty value from the values received,
    so it's typically used as an output processor to single-valued fields.
    It doesn't receive any ``__init__`` method arguments, nor does it accept Loader contexts.

    Example:

    >>> from itemloaders.processors import TakeFirst
    >>> proc = TakeFirst()
    >>> proc(['', 'one', 'two', 'three'])
    'one'
    """

    def __call__(self, values):
        for value in values:
            if value is not None and value != '':
                return value