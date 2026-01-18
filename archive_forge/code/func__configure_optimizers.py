from contextlib import contextmanager
from dataclasses import fields
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union, overload
from weakref import proxy
import torch
from torch import optim
from torch.optim import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.types import Optimizable, ReduceLROnPlateau, _Stateful
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.types import LRSchedulerConfig, LRSchedulerTypeTuple
def _configure_optimizers(optim_conf: Union[Dict[str, Any], List, Optimizer, Tuple]) -> Tuple[List, List, Optional[str]]:
    optimizers, lr_schedulers = ([], [])
    monitor = None
    if isinstance(optim_conf, Optimizable):
        optimizers = [optim_conf]
    elif isinstance(optim_conf, (list, tuple)) and len(optim_conf) == 2 and isinstance(optim_conf[0], list) and all((isinstance(opt, Optimizable) for opt in optim_conf[0])):
        opt, sch = optim_conf
        optimizers = opt
        lr_schedulers = sch if isinstance(sch, list) else [sch]
    elif isinstance(optim_conf, dict):
        _validate_optim_conf(optim_conf)
        optimizers = [optim_conf['optimizer']]
        monitor = optim_conf.get('monitor', None)
        lr_schedulers = [optim_conf['lr_scheduler']] if 'lr_scheduler' in optim_conf else []
    elif isinstance(optim_conf, (list, tuple)) and all((isinstance(d, dict) for d in optim_conf)):
        for opt_dict in optim_conf:
            _validate_optim_conf(opt_dict)
        optimizers = [opt_dict['optimizer'] for opt_dict in optim_conf]
        scheduler_dict = lambda scheduler: dict(scheduler) if isinstance(scheduler, dict) else {'scheduler': scheduler}
        lr_schedulers = [scheduler_dict(opt_dict['lr_scheduler']) for opt_dict in optim_conf if 'lr_scheduler' in opt_dict]
    elif isinstance(optim_conf, (list, tuple)) and all((isinstance(opt, Optimizable) for opt in optim_conf)):
        optimizers = list(optim_conf)
    else:
        raise MisconfigurationException('Unknown configuration for model optimizers. Output from `model.configure_optimizers()` should be one of:\n * `Optimizer`\n * [`Optimizer`]\n * ([`Optimizer`], [`LRScheduler`])\n * {"optimizer": `Optimizer`, (optional) "lr_scheduler": `LRScheduler`}\n')
    return (optimizers, lr_schedulers, monitor)