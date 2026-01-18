from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class ValueDefWithConditionStringFieldDefText(TextDef):
    """ValueDefWithConditionStringFieldDefText schema wrapper

    Parameters
    ----------

    condition : dict, :class:`ConditionalStringFieldDef`, :class:`ConditionalValueDefTextExprRef`, :class:`ConditionalParameterStringFieldDef`, :class:`ConditionalPredicateStringFieldDef`, :class:`ConditionalParameterValueDefTextExprRef`, :class:`ConditionalPredicateValueDefTextExprRef`, Sequence[dict, :class:`ConditionalValueDefTextExprRef`, :class:`ConditionalParameterValueDefTextExprRef`, :class:`ConditionalPredicateValueDefTextExprRef`]
        A field definition or one or more value definition(s) with a parameter predicate.
    value : str, dict, :class:`Text`, Sequence[str], :class:`ExprRef`
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ValueDefWithCondition<StringFieldDef,Text>'}

    def __init__(self, condition: Union[dict, 'SchemaBase', Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, value: Union[str, dict, '_Parameter', 'SchemaBase', Sequence[str], UndefinedType]=Undefined, **kwds):
        super(ValueDefWithConditionStringFieldDefText, self).__init__(condition=condition, value=value, **kwds)