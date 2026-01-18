from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class WindowFieldDef(VegaLiteSchema):
    """WindowFieldDef schema wrapper

    Parameters
    ----------

    op : :class:`AggregateOp`, :class:`WindowOnlyOp`, Literal['row_number', 'rank', 'dense_rank', 'percent_rank', 'cume_dist', 'ntile', 'lag', 'lead', 'first_value', 'last_value', 'nth_value'], Literal['argmax', 'argmin', 'average', 'count', 'distinct', 'max', 'mean', 'median', 'min', 'missing', 'product', 'q1', 'q3', 'ci0', 'ci1', 'stderr', 'stdev', 'stdevp', 'sum', 'valid', 'values', 'variance', 'variancep', 'exponential', 'exponentialb']
        The window or aggregation operation to apply within a window (e.g., ``"rank"``,
        ``"lead"``, ``"sum"``, ``"average"`` or ``"count"`` ). See the list of all supported
        operations `here <https://vega.github.io/vega-lite/docs/window.html#ops>`__.
    field : str, :class:`FieldName`
        The data field for which to compute the aggregate or window function. This can be
        omitted for window functions that do not operate over a field such as ``"count"``,
        ``"rank"``, ``"dense_rank"``.
    param : float
        Parameter values for the window functions. Parameter values can be omitted for
        operations that do not accept a parameter.

        See the list of all supported operations and their parameters `here
        <https://vega.github.io/vega-lite/docs/transforms/window.html>`__.
    as : str, :class:`FieldName`
        The output name for the window operation.
    """
    _schema = {'$ref': '#/definitions/WindowFieldDef'}

    def __init__(self, op: Union['SchemaBase', Literal['row_number', 'rank', 'dense_rank', 'percent_rank', 'cume_dist', 'ntile', 'lag', 'lead', 'first_value', 'last_value', 'nth_value'], Literal['argmax', 'argmin', 'average', 'count', 'distinct', 'max', 'mean', 'median', 'min', 'missing', 'product', 'q1', 'q3', 'ci0', 'ci1', 'stderr', 'stdev', 'stdevp', 'sum', 'valid', 'values', 'variance', 'variancep', 'exponential', 'exponentialb'], UndefinedType]=Undefined, field: Union[str, 'SchemaBase', UndefinedType]=Undefined, param: Union[float, UndefinedType]=Undefined, **kwds):
        super(WindowFieldDef, self).__init__(op=op, field=field, param=param, **kwds)