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
def _lightning_get_all_attr_holders(model: 'pl.LightningModule', attribute: str) -> List[Any]:
    """Special attribute finding for Lightning.

    Gets all of the objects or dicts that holds attribute. Checks for attribute in model namespace, the old hparams
    namespace/dict, and the datamodule.

    """
    holders: List[Any] = []
    if hasattr(model, attribute):
        holders.append(model)
    if hasattr(model, 'hparams') and attribute in model.hparams:
        holders.append(model.hparams)
    trainer = model._trainer
    if trainer is not None and trainer.datamodule is not None:
        if hasattr(trainer.datamodule, attribute):
            holders.append(trainer.datamodule)
        if hasattr(trainer.datamodule, 'hparams') and attribute in trainer.datamodule.hparams:
            holders.append(trainer.datamodule.hparams)
    return holders