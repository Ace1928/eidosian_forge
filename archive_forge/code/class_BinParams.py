from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class BinParams(VegaLiteSchema):
    """BinParams schema wrapper
    Binning properties or boolean flag for determining whether to bin data or not.

    Parameters
    ----------

    anchor : float
        A value in the binned domain at which to anchor the bins, shifting the bin
        boundaries if necessary to ensure that a boundary aligns with the anchor value.

        **Default value:** the minimum bin extent value
    base : float
        The number base to use for automatic bin determination (default is base 10).

        **Default value:** ``10``
    binned : bool
        When set to ``true``, Vega-Lite treats the input data as already binned.
    divide : Sequence[float]
        Scale factors indicating allowable subdivisions. The default value is [5, 2], which
        indicates that for base 10 numbers (the default base), the method may consider
        dividing bin sizes by 5 and/or 2. For example, for an initial step size of 10, the
        method can check if bin sizes of 2 (= 10/5), 5 (= 10/2), or 1 (= 10/(5*2)) might
        also satisfy the given constraints.

        **Default value:** ``[5, 2]``
    extent : dict, Sequence[float], :class:`BinExtent`, :class:`ParameterExtent`
        A two-element ( ``[min, max]`` ) array indicating the range of desired bin values.
    maxbins : float
        Maximum number of bins.

        **Default value:** ``6`` for ``row``, ``column`` and ``shape`` channels; ``10`` for
        other channels
    minstep : float
        A minimum allowable step size (particularly useful for integer values).
    nice : bool
        If true, attempts to make the bin boundaries use human-friendly boundaries, such as
        multiples of ten.

        **Default value:** ``true``
    step : float
        An exact step size to use between bins.

        **Note:** If provided, options such as maxbins will be ignored.
    steps : Sequence[float]
        An array of allowable step sizes to choose from.
    """
    _schema = {'$ref': '#/definitions/BinParams'}

    def __init__(self, anchor: Union[float, UndefinedType]=Undefined, base: Union[float, UndefinedType]=Undefined, binned: Union[bool, UndefinedType]=Undefined, divide: Union[Sequence[float], UndefinedType]=Undefined, extent: Union[dict, '_Parameter', 'SchemaBase', Sequence[float], UndefinedType]=Undefined, maxbins: Union[float, UndefinedType]=Undefined, minstep: Union[float, UndefinedType]=Undefined, nice: Union[bool, UndefinedType]=Undefined, step: Union[float, UndefinedType]=Undefined, steps: Union[Sequence[float], UndefinedType]=Undefined, **kwds):
        super(BinParams, self).__init__(anchor=anchor, base=base, binned=binned, divide=divide, extent=extent, maxbins=maxbins, minstep=minstep, nice=nice, step=step, steps=steps, **kwds)