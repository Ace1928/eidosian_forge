from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class ValueDefWithConditionMarkPropFieldOrDatumDefnumberArray(MarkPropDefnumberArray, NumericArrayMarkPropDef):
    """ValueDefWithConditionMarkPropFieldOrDatumDefnumberArray schema wrapper

    Parameters
    ----------

    condition : dict, :class:`ConditionalMarkPropFieldOrDatumDef`, :class:`ConditionalValueDefnumberArrayExprRef`, :class:`ConditionalParameterMarkPropFieldOrDatumDef`, :class:`ConditionalPredicateMarkPropFieldOrDatumDef`, :class:`ConditionalParameterValueDefnumberArrayExprRef`, :class:`ConditionalPredicateValueDefnumberArrayExprRef`, Sequence[dict, :class:`ConditionalValueDefnumberArrayExprRef`, :class:`ConditionalParameterValueDefnumberArrayExprRef`, :class:`ConditionalPredicateValueDefnumberArrayExprRef`]
        A field definition or one or more value definition(s) with a parameter predicate.
    value : dict, Sequence[float], :class:`ExprRef`
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ValueDefWithCondition<MarkPropFieldOrDatumDef,number[]>'}

    def __init__(self, condition: Union[dict, 'SchemaBase', Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, value: Union[dict, '_Parameter', 'SchemaBase', Sequence[float], UndefinedType]=Undefined, **kwds):
        super(ValueDefWithConditionMarkPropFieldOrDatumDefnumberArray, self).__init__(condition=condition, value=value, **kwds)