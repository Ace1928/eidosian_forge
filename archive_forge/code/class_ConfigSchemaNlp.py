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
class ConfigSchemaNlp(BaseModel):
    lang: StrictStr = Field(..., title='The base language to use')
    pipeline: List[StrictStr] = Field(..., title='The pipeline component names in order')
    disabled: List[StrictStr] = Field(..., title='Pipeline components to disable by default')
    tokenizer: Callable = Field(..., title='The tokenizer to use')
    before_creation: Optional[Callable[[Type['Language']], Type['Language']]] = Field(..., title='Optional callback to modify Language class before initialization')
    after_creation: Optional[Callable[['Language'], 'Language']] = Field(..., title='Optional callback to modify nlp object after creation and before the pipeline is constructed')
    after_pipeline_creation: Optional[Callable[['Language'], 'Language']] = Field(..., title='Optional callback to modify nlp object after the pipeline is constructed')
    batch_size: Optional[int] = Field(..., title='Default batch size')
    vectors: Callable = Field(..., title='Vectors implementation')

    class Config:
        extra = 'forbid'
        arbitrary_types_allowed = True