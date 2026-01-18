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
def _lightning_get_first_attr_holder(model: 'pl.LightningModule', attribute: str) -> Optional[Any]:
    """Special attribute finding for Lightning.

    Gets the object or dict that holds attribute, or None. Checks for attribute in model namespace, the old hparams
    namespace/dict, and the datamodule, returns the last one that has it.

    """
    holders = _lightning_get_all_attr_holders(model, attribute)
    if len(holders) == 0:
        return None
    return holders[-1]