import copy
import inspect
import pickle
import types
from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Literal, MutableMapping, Optional, Sequence, Tuple, Type, Union
from torch import nn
import pytorch_lightning as pl
from lightning_fabric.utilities.data import AttributeDict as _AttributeDict
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def _get_first_if_any(params: List[inspect.Parameter], param_type: Literal[inspect._ParameterKind.VAR_POSITIONAL, inspect._ParameterKind.VAR_KEYWORD]) -> Optional[str]:
    for p in params:
        if p.kind == param_type:
            return p.name
    return None