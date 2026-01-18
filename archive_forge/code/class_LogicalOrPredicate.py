from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class LogicalOrPredicate(PredicateComposition):
    """LogicalOrPredicate schema wrapper

    Parameters
    ----------

    or : Sequence[str, dict, :class:`Predicate`, :class:`FieldGTPredicate`, :class:`FieldLTPredicate`, :class:`FieldGTEPredicate`, :class:`FieldLTEPredicate`, :class:`LogicalOrPredicate`, :class:`ParameterPredicate`, :class:`FieldEqualPredicate`, :class:`FieldOneOfPredicate`, :class:`FieldRangePredicate`, :class:`FieldValidPredicate`, :class:`LogicalAndPredicate`, :class:`LogicalNotPredicate`, :class:`PredicateComposition`]

    """
    _schema = {'$ref': '#/definitions/LogicalOr<Predicate>'}

    def __init__(self, **kwds):
        super(LogicalOrPredicate, self).__init__(**kwds)