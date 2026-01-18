from collections import ChainMap
from itemloaders.common import wrap_loader_context
from itemloaders.utils import arg_to_iter
class MapCompose:
    """
    A processor which is constructed from the composition of the given
    functions, similar to the :class:`Compose` processor. The difference with
    this processor is the way internal results are passed among functions,
    which is as follows:

    The input value of this processor is *iterated* and the first function is
    applied to each element. The results of these function calls (one for each element)
    are concatenated to construct a new iterable, which is then used to apply the
    second function, and so on, until the last function is applied to each
    value of the list of values collected so far. The output values of the last
    function are concatenated together to produce the output of this processor.

    Each particular function can return a value or a list of values, which is
    flattened with the list of values returned by the same function applied to
    the other input values. The functions can also return ``None`` in which
    case the output of that function is ignored for further processing over the
    chain.

    This processor provides a convenient way to compose functions that only
    work with single values (instead of iterables). For this reason the
    :class:`MapCompose` processor is typically used as input processor, since
    data is often extracted using the
    :meth:`~parsel.selector.Selector.extract` method of `parsel selectors`_,
    which returns a list of unicode strings.

    The example below should clarify how it works:

    >>> def filter_world(x):
    ...     return None if x == 'world' else x
    ...
    >>> from itemloaders.processors import MapCompose
    >>> proc = MapCompose(filter_world, str.upper)
    >>> proc(['hello', 'world', 'this', 'is', 'something'])
    ['HELLO', 'THIS', 'IS', 'SOMETHING']

    As with the Compose processor, functions can receive Loader contexts, and
    ``__init__`` method keyword arguments are used as default context values.
    See :class:`Compose` processor for more info.

    .. _`parsel selectors`: https://parsel.readthedocs.io/en/latest/parsel.html#parsel.selector.Selector.extract
    """

    def __init__(self, *functions, **default_loader_context):
        self.functions = functions
        self.default_loader_context = default_loader_context

    def __call__(self, value, loader_context=None):
        values = arg_to_iter(value)
        if loader_context:
            context = ChainMap(loader_context, self.default_loader_context)
        else:
            context = self.default_loader_context
        wrapped_funcs = [wrap_loader_context(f, context) for f in self.functions]
        for func in wrapped_funcs:
            next_values = []
            for v in values:
                try:
                    next_values += arg_to_iter(func(v))
                except Exception as e:
                    raise ValueError("Error in MapCompose with %s value=%r error='%s: %s'" % (str(func), value, type(e).__name__, str(e))) from e
            values = next_values
        return values