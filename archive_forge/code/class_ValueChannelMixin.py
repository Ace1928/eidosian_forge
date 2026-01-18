import sys
from . import core
import pandas as pd
from altair.utils.schemapi import Undefined, UndefinedType, with_property_setters
from altair.utils import parse_shorthand
from typing import Any, overload, Sequence, List, Literal, Union, Optional
from typing import Dict as TypingDict
class ValueChannelMixin:

    def to_dict(self, validate: bool=True, ignore: Optional[List[str]]=None, context: Optional[TypingDict[str, Any]]=None) -> dict:
        context = context or {}
        ignore = ignore or []
        condition = self._get('condition', Undefined)
        copy = self
        if condition is not Undefined:
            if isinstance(condition, core.SchemaBase):
                pass
            elif 'field' in condition and 'type' not in condition:
                kwds = parse_shorthand(condition['field'], context.get('data', None))
                copy = self.copy(deep=['condition'])
                copy['condition'].update(kwds)
        return super(ValueChannelMixin, copy).to_dict(validate=validate, ignore=ignore, context=context)