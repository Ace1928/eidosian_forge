import inspect
import pickle
from types import GeneratorType
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
import catalogue
import numpy
import pytest
import thinc.config
from thinc.api import Config, Model, NumpyOps, RAdam
from thinc.config import ConfigValidationError
from thinc.types import Generator, Ragged
from thinc.util import partial
from .util import make_tempdir
class ComplexSchema(BaseModel):
    outer_req: int
    outer_opt: str = 'default value'
    level2_req: HelloIntsSchema
    level2_opt: DefaultsSchema = DefaultsSchema(required=1)