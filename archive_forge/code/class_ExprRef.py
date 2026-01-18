from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class ExprRef(VegaLiteSchema):
    """ExprRef schema wrapper

    Parameters
    ----------

    expr : str
        Vega expression (which can refer to Vega-Lite parameters).
    """
    _schema = {'$ref': '#/definitions/ExprRef'}

    def __init__(self, expr: Union[str, UndefinedType]=Undefined, **kwds):
        super(ExprRef, self).__init__(expr=expr, **kwds)