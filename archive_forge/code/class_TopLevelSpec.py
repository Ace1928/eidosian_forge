from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class TopLevelSpec(VegaLiteSchema):
    """TopLevelSpec schema wrapper
    A Vega-Lite top-level specification. This is the root class for all Vega-Lite
    specifications. (The json schema is generated from this type.)
    """
    _schema = {'$ref': '#/definitions/TopLevelSpec'}

    def __init__(self, *args, **kwds):
        super(TopLevelSpec, self).__init__(*args, **kwds)