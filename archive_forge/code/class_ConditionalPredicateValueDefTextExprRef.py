from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class ConditionalPredicateValueDefTextExprRef(ConditionalValueDefTextExprRef):
    """ConditionalPredicateValueDefTextExprRef schema wrapper

    Parameters
    ----------

    test : str, dict, :class:`Predicate`, :class:`FieldGTPredicate`, :class:`FieldLTPredicate`, :class:`FieldGTEPredicate`, :class:`FieldLTEPredicate`, :class:`LogicalOrPredicate`, :class:`ParameterPredicate`, :class:`FieldEqualPredicate`, :class:`FieldOneOfPredicate`, :class:`FieldRangePredicate`, :class:`FieldValidPredicate`, :class:`LogicalAndPredicate`, :class:`LogicalNotPredicate`, :class:`PredicateComposition`
        Predicate for triggering the condition
    value : str, dict, :class:`Text`, Sequence[str], :class:`ExprRef`
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ConditionalPredicate<ValueDef<(Text|ExprRef)>>'}

    def __init__(self, test: Union[str, dict, 'SchemaBase', UndefinedType]=Undefined, value: Union[str, dict, '_Parameter', 'SchemaBase', Sequence[str], UndefinedType]=Undefined, **kwds):
        super(ConditionalPredicateValueDefTextExprRef, self).__init__(test=test, value=value, **kwds)