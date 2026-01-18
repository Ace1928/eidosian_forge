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
class ConfigSchemaInit(BaseModel):
    vocab_data: Optional[StrictStr] = Field(..., title='Path to JSON-formatted vocabulary file')
    lookups: Optional[Lookups] = Field(..., title='Vocabulary lookups, e.g. lexeme normalization')
    vectors: Optional[StrictStr] = Field(..., title='Path to vectors')
    init_tok2vec: Optional[StrictStr] = Field(..., title='Path to pretrained tok2vec weights')
    tokenizer: Dict[StrictStr, Any] = Field(..., help='Arguments to be passed into Tokenizer.initialize')
    components: Dict[StrictStr, Dict[StrictStr, Any]] = Field(..., help='Arguments for TrainablePipe.initialize methods of pipeline components, keyed by component')
    before_init: Optional[Callable[['Language'], 'Language']] = Field(..., title='Optional callback to modify nlp object before initialization')
    after_init: Optional[Callable[['Language'], 'Language']] = Field(..., title='Optional callback to modify nlp object after initialization')

    class Config:
        extra = 'forbid'
        arbitrary_types_allowed = True