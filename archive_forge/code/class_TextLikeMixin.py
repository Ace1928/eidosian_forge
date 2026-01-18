import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
class TextLikeMixin:

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        data['children'] = [c.model_dump() for c in self.children]
        return data

    @classmethod
    def model_validate(cls, data):
        d = deepcopy(data)
        children = []
        for c in d.get('children'):
            _type = c.get('type')
            if _type == 'link':
                child = InlineLink.model_validate(c)
            elif _type == 'latex':
                child = InlineLatex.model_validate(c)
            elif _type == 'paragraph':
                child = Paragraph.model_validate(c)
            else:
                child = Text.model_validate(c)
            children.append(child)
        d['children'] = children
        obj = cls(**d)
        return obj