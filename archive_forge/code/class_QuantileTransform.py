from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class QuantileTransform(Transform):
    """QuantileTransform schema wrapper

    Parameters
    ----------

    quantile : str, :class:`FieldName`
        The data field for which to perform quantile estimation.
    groupby : Sequence[str, :class:`FieldName`]
        The data fields to group by. If not specified, a single group containing all data
        objects will be used.
    probs : Sequence[float]
        An array of probabilities in the range (0, 1) for which to compute quantile values.
        If not specified, the *step* parameter will be used.
    step : float
        A probability step size (default 0.01) for sampling quantile values. All values from
        one-half the step size up to 1 (exclusive) will be sampled. This parameter is only
        used if the *probs* parameter is not provided.
    as : Sequence[str, :class:`FieldName`]
        The output field names for the probability and quantile values.

        **Default value:** ``["prob", "value"]``
    """
    _schema = {'$ref': '#/definitions/QuantileTransform'}

    def __init__(self, quantile: Union[str, 'SchemaBase', UndefinedType]=Undefined, groupby: Union[Sequence[Union[str, 'SchemaBase']], UndefinedType]=Undefined, probs: Union[Sequence[float], UndefinedType]=Undefined, step: Union[float, UndefinedType]=Undefined, **kwds):
        super(QuantileTransform, self).__init__(quantile=quantile, groupby=groupby, probs=probs, step=step, **kwds)