import sys
from . import core
import pandas as pd
from altair.utils.schemapi import Undefined, UndefinedType, with_property_setters
from altair.utils import parse_shorthand
from typing import Any, overload, Sequence, List, Literal, Union, Optional
from typing import Dict as TypingDict
class FieldChannelMixin:

    def to_dict(self, validate: bool=True, ignore: Optional[List[str]]=None, context: Optional[TypingDict[str, Any]]=None) -> Union[dict, List[dict]]:
        context = context or {}
        ignore = ignore or []
        shorthand = self._get('shorthand')
        field = self._get('field')
        if shorthand is not Undefined and field is not Undefined:
            raise ValueError('{} specifies both shorthand={} and field={}. '.format(self.__class__.__name__, shorthand, field))
        if isinstance(shorthand, (tuple, list)):
            kwds = self._kwds.copy()
            kwds.pop('shorthand')
            return [self.__class__(sh, **kwds).to_dict(validate=validate, ignore=ignore, context=context) for sh in shorthand]
        if shorthand is Undefined:
            parsed = {}
        elif isinstance(shorthand, str):
            parsed = parse_shorthand(shorthand, data=context.get('data', None))
            type_required = 'type' in self._kwds
            type_in_shorthand = 'type' in parsed
            type_defined_explicitly = self._get('type') is not Undefined
            if not type_required:
                parsed.pop('type', None)
            elif not (type_in_shorthand or type_defined_explicitly):
                if isinstance(context.get('data', None), pd.DataFrame):
                    raise ValueError('Unable to determine data type for the field "{}"; verify that the field name is not misspelled. If you are referencing a field from a transform, also confirm that the data type is specified correctly.'.format(shorthand))
                else:
                    raise ValueError('{} encoding field is specified without a type; the type cannot be automatically inferred because the data is not specified as a pandas.DataFrame.'.format(shorthand))
        else:
            parsed = {'field': shorthand}
        context['parsed_shorthand'] = parsed
        return super(FieldChannelMixin, self).to_dict(validate=validate, ignore=ignore, context=context)