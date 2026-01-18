from pathlib import Path
from typing import Any, List, Tuple, Union
from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Checkpoint
def _log_hyperparams(trainer: 'pl.Trainer') -> None:
    if not trainer.loggers:
        return
    pl_module = trainer.lightning_module
    datamodule_log_hyperparams = trainer.datamodule._log_hyperparams if trainer.datamodule is not None else False
    hparams_initial = None
    if pl_module._log_hyperparams and datamodule_log_hyperparams:
        datamodule_hparams = trainer.datamodule.hparams_initial
        lightning_hparams = pl_module.hparams_initial
        inconsistent_keys = []
        for key in lightning_hparams.keys() & datamodule_hparams.keys():
            lm_val, dm_val = (lightning_hparams[key], datamodule_hparams[key])
            if type(lm_val) != type(dm_val) or (isinstance(lm_val, Tensor) and id(lm_val) != id(dm_val)) or lm_val != dm_val:
                inconsistent_keys.append(key)
        if inconsistent_keys:
            raise RuntimeError(f"Error while merging hparams: the keys {inconsistent_keys} are present in both the LightningModule's and LightningDataModule's hparams but have different values.")
        hparams_initial = {**lightning_hparams, **datamodule_hparams}
    elif pl_module._log_hyperparams:
        hparams_initial = pl_module.hparams_initial
    elif datamodule_log_hyperparams:
        hparams_initial = trainer.datamodule.hparams_initial
    for logger in trainer.loggers:
        if hparams_initial is not None:
            logger.log_hyperparams(hparams_initial)
        logger.log_graph(pl_module)
        logger.save()