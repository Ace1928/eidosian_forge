from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class JoinAggregateFieldDef(VegaLiteSchema):
    """JoinAggregateFieldDef schema wrapper

    Parameters
    ----------

    op : :class:`AggregateOp`, Literal['argmax', 'argmin', 'average', 'count', 'distinct', 'max', 'mean', 'median', 'min', 'missing', 'product', 'q1', 'q3', 'ci0', 'ci1', 'stderr', 'stdev', 'stdevp', 'sum', 'valid', 'values', 'variance', 'variancep', 'exponential', 'exponentialb']
        The aggregation operation to apply (e.g., ``"sum"``, ``"average"`` or ``"count"`` ).
        See the list of all supported operations `here
        <https://vega.github.io/vega-lite/docs/aggregate.html#ops>`__.
    field : str, :class:`FieldName`
        The data field for which to compute the aggregate function. This can be omitted for
        functions that do not operate over a field such as ``"count"``.
    as : str, :class:`FieldName`
        The output name for the join aggregate operation.
    """
    _schema = {'$ref': '#/definitions/JoinAggregateFieldDef'}

    def __init__(self, op: Union['SchemaBase', Literal['argmax', 'argmin', 'average', 'count', 'distinct', 'max', 'mean', 'median', 'min', 'missing', 'product', 'q1', 'q3', 'ci0', 'ci1', 'stderr', 'stdev', 'stdevp', 'sum', 'valid', 'values', 'variance', 'variancep', 'exponential', 'exponentialb'], UndefinedType]=Undefined, field: Union[str, 'SchemaBase', UndefinedType]=Undefined, **kwds):
        super(JoinAggregateFieldDef, self).__init__(op=op, field=field, **kwds)