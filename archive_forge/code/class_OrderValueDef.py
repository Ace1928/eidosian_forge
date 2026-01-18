from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class OrderValueDef(VegaLiteSchema):
    """OrderValueDef schema wrapper

    Parameters
    ----------

    value : dict, float, :class:`ExprRef`
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    condition : dict, :class:`ConditionalValueDefnumber`, :class:`ConditionalParameterValueDefnumber`, :class:`ConditionalPredicateValueDefnumber`, Sequence[dict, :class:`ConditionalValueDefnumber`, :class:`ConditionalParameterValueDefnumber`, :class:`ConditionalPredicateValueDefnumber`]
        One or more value definition(s) with `a parameter or a test predicate
        <https://vega.github.io/vega-lite/docs/condition.html>`__.

        **Note:** A field definition's ``condition`` property can only contain `conditional
        value definitions <https://vega.github.io/vega-lite/docs/condition.html#value>`__
        since Vega-Lite only allows at most one encoded field per encoding channel.
    """
    _schema = {'$ref': '#/definitions/OrderValueDef'}

    def __init__(self, value: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, condition: Union[dict, 'SchemaBase', Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, **kwds):
        super(OrderValueDef, self).__init__(value=value, condition=condition, **kwds)