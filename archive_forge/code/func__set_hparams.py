import copy
import inspect
import types
from argparse import Namespace
from typing import Any, List, MutableMapping, Optional, Sequence, Union
from lightning_fabric.utilities.data import AttributeDict
from pytorch_lightning.utilities.parsing import save_hyperparameters
def _set_hparams(self, hp: Union[MutableMapping, Namespace, str]) -> None:
    hp = self._to_hparams_dict(hp)
    if isinstance(hp, dict) and isinstance(self.hparams, dict):
        self.hparams.update(hp)
    else:
        self._hparams = hp