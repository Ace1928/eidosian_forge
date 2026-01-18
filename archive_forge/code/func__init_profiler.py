from typing import Optional, Union
import pytorch_lightning as pl
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.accelerators import CUDAAccelerator, MPSAccelerator, XLAAccelerator
from pytorch_lightning.loggers.logger import DummyLogger
from pytorch_lightning.profilers import (
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _graphcore_available_and_importable, _habana_available_and_importable
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
def _init_profiler(trainer: 'pl.Trainer', profiler: Optional[Union[Profiler, str]]) -> None:
    if isinstance(profiler, str):
        PROFILERS = {'simple': SimpleProfiler, 'advanced': AdvancedProfiler, 'pytorch': PyTorchProfiler, 'xla': XLAProfiler}
        profiler = profiler.lower()
        if profiler not in PROFILERS:
            raise MisconfigurationException(f'When passing string value for the `profiler` parameter of `Trainer`, it can only be one of {list(PROFILERS.keys())}')
        profiler_class = PROFILERS[profiler]
        profiler = profiler_class()
    trainer.profiler = profiler or PassThroughProfiler()