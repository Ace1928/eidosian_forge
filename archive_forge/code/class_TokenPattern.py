import inspect
import re
from collections import defaultdict
from enum import Enum
from typing import (
from thinc.api import ConfigValidationError, Model, Optimizer
from thinc.config import Promise
from .attrs import NAMES
from .compat import Literal
from .lookups import Lookups
from .util import is_cython_func
class TokenPattern(BaseModel):
    orth: Optional[StringValue] = None
    text: Optional[StringValue] = None
    lower: Optional[StringValue] = None
    pos: Optional[StringValue] = None
    tag: Optional[StringValue] = None
    morph: Optional[StringValue] = None
    dep: Optional[StringValue] = None
    lemma: Optional[StringValue] = None
    shape: Optional[StringValue] = None
    ent_type: Optional[StringValue] = None
    ent_iob: Optional[IobValue] = None
    ent_id: Optional[StringValue] = None
    ent_kb_id: Optional[StringValue] = None
    norm: Optional[StringValue] = None
    length: Optional[NumberValue] = None
    spacy: Optional[StrictBool] = None
    is_alpha: Optional[StrictBool] = None
    is_ascii: Optional[StrictBool] = None
    is_digit: Optional[StrictBool] = None
    is_lower: Optional[StrictBool] = None
    is_upper: Optional[StrictBool] = None
    is_title: Optional[StrictBool] = None
    is_punct: Optional[StrictBool] = None
    is_space: Optional[StrictBool] = None
    is_bracket: Optional[StrictBool] = None
    is_quote: Optional[StrictBool] = None
    is_left_punct: Optional[StrictBool] = None
    is_right_punct: Optional[StrictBool] = None
    is_currency: Optional[StrictBool] = None
    is_stop: Optional[StrictBool] = None
    is_sent_start: Optional[StrictBool] = None
    sent_start: Optional[StrictBool] = None
    like_num: Optional[StrictBool] = None
    like_url: Optional[StrictBool] = None
    like_email: Optional[StrictBool] = None
    op: Optional[TokenPatternOperator] = None
    underscore: Optional[Dict[StrictStr, UnderscoreValue]] = Field(None, alias='_')

    class Config:
        extra = 'forbid'
        allow_population_by_field_name = True
        alias_generator = lambda value: value.upper()

    @validator('*', pre=True, allow_reuse=True)
    def raise_for_none(cls, v):
        if v is None:
            raise ValueError('None / null is not allowed')
        return v