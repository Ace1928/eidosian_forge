from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class AggregatedFieldDef(VegaLiteSchema):
    """AggregatedFieldDef schema wrapper

    Parameters
    ----------

    op : :class:`AggregateOp`, Literal['argmax', 'argmin', 'average', 'count', 'distinct', 'max', 'mean', 'median', 'min', 'missing', 'product', 'q1', 'q3', 'ci0', 'ci1', 'stderr', 'stdev', 'stdevp', 'sum', 'valid', 'values', 'variance', 'variancep', 'exponential', 'exponentialb']
        The aggregation operation to apply to the fields (e.g., ``"sum"``, ``"average"``, or
        ``"count"`` ). See the `full list of supported aggregation operations
        <https://vega.github.io/vega-lite/docs/aggregate.html#ops>`__ for more information.
    field : str, :class:`FieldName`
        The data field for which to compute aggregate function. This is required for all
        aggregation operations except ``"count"``.
    as : str, :class:`FieldName`
        The output field names to use for each aggregated field.
    """
    _schema = {'$ref': '#/definitions/AggregatedFieldDef'}

    def __init__(self, op: Union['SchemaBase', Literal['argmax', 'argmin', 'average', 'count', 'distinct', 'max', 'mean', 'median', 'min', 'missing', 'product', 'q1', 'q3', 'ci0', 'ci1', 'stderr', 'stdev', 'stdevp', 'sum', 'valid', 'values', 'variance', 'variancep', 'exponential', 'exponentialb'], UndefinedType]=Undefined, field: Union[str, 'SchemaBase', UndefinedType]=Undefined, **kwds):
        super(AggregatedFieldDef, self).__init__(op=op, field=field, **kwds)