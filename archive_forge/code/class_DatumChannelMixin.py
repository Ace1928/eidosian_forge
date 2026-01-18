import sys
from . import core
import pandas as pd
from altair.utils.schemapi import Undefined, UndefinedType, with_property_setters
from altair.utils import parse_shorthand
from typing import Any, overload, Sequence, List, Literal, Union, Optional
from typing import Dict as TypingDict
class DatumChannelMixin:

    def to_dict(self, validate: bool=True, ignore: Optional[List[str]]=None, context: Optional[TypingDict[str, Any]]=None) -> dict:
        context = context or {}
        ignore = ignore or []
        datum = self._get('datum', Undefined)
        copy = self
        if datum is not Undefined:
            if isinstance(datum, core.SchemaBase):
                pass
        return super(DatumChannelMixin, copy).to_dict(validate=validate, ignore=ignore, context=context)