import json
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Sequence, Tuple, Type, Union
from .json import pydantic_encoder
from .utils import Representation
def _display_error_type_and_ctx(error: 'ErrorDict') -> str:
    t = 'type=' + error['type']
    ctx = error.get('ctx')
    if ctx:
        return t + ''.join((f'; {k}={v}' for k, v in ctx.items()))
    else:
        return t