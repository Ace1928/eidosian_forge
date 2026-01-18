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
class ConfigSchemaPretrain(BaseModel):
    max_epochs: StrictInt = Field(..., title='Maximum number of epochs to train for')
    dropout: StrictFloat = Field(..., title='Dropout rate')
    n_save_every: Optional[StrictInt] = Field(..., title='Saving additional temporary model after n batches within an epoch')
    n_save_epoch: Optional[StrictInt] = Field(..., title='Saving model after every n epoch')
    optimizer: Optimizer = Field(..., title='The optimizer to use')
    corpus: StrictStr = Field(..., title='Path in the config to the training data')
    batcher: Batcher = Field(..., title='Batcher for the training data')
    component: str = Field(..., title='Component to find the layer to pretrain')
    layer: str = Field(..., title='Layer to pretrain. Whole model if empty.')
    objective: Callable[['Vocab', Model], Model] = Field(..., title='A function that creates the pretraining objective.')

    class Config:
        extra = 'forbid'
        arbitrary_types_allowed = True