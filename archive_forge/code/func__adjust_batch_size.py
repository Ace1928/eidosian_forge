import logging
import os
import uuid
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple
import pytorch_lightning as pl
from pytorch_lightning.utilities.memory import garbage_collection_cuda, is_oom_error
from pytorch_lightning.utilities.parsing import lightning_getattr, lightning_setattr
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
def _adjust_batch_size(trainer: 'pl.Trainer', batch_arg_name: str='batch_size', factor: float=1.0, value: Optional[int]=None, desc: Optional[str]=None) -> Tuple[int, bool]:
    """Helper function for adjusting the batch size.

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

    """
    model = trainer.lightning_module
    batch_size = lightning_getattr(model, batch_arg_name)
    assert batch_size is not None
    loop = trainer._active_loop
    assert loop is not None
    loop.setup_data()
    combined_loader = loop._combined_loader
    assert combined_loader is not None
    try:
        combined_dataset_length = combined_loader._dataset_length()
        if batch_size >= combined_dataset_length:
            rank_zero_info(f'The batch size {batch_size} is greater or equal than the length of your dataset.')
            return (batch_size, False)
    except NotImplementedError:
        pass
    new_size = value if value is not None else int(batch_size * factor)
    if desc:
        rank_zero_info(f'Batch size {batch_size} {desc}, trying batch size {new_size}')
    changed = new_size != batch_size
    lightning_setattr(model, batch_arg_name, new_size)
    return (new_size, changed)