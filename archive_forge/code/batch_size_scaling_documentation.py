import logging
import os
import uuid
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple
import pytorch_lightning as pl
from pytorch_lightning.utilities.memory import garbage_collection_cuda, is_oom_error
from pytorch_lightning.utilities.parsing import lightning_getattr, lightning_setattr
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
Helper function for adjusting the batch size.

    Args:
        trainer: instance of pytorch_lightning.Trainer
        factor: value which the old batch size is multiplied by to get the
            new batch size
        value: if a value is given, will override the batch size with this value.
            Note that the value of `factor` will not have an effect in this case
        desc: either ``"succeeded"`` or ``"failed"``. Used purely for logging

    Returns:
        The new batch size for the next trial and a bool that signals whether the
        new value is different than the previous batch size.

    