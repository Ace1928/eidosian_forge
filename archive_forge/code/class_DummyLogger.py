import functools
import operator
from abc import ABC
from collections import defaultdict
from typing import Any, Callable, Dict, Mapping, Optional, Sequence
import numpy as np
from typing_extensions import override
from lightning_fabric.loggers import Logger as FabricLogger
from lightning_fabric.loggers.logger import _DummyExperiment as DummyExperiment  # for backward compatibility
from lightning_fabric.loggers.logger import rank_zero_experiment  # noqa: F401  # for backward compatibility
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
class DummyLogger(Logger):
    """Dummy logger for internal use.

    It is useful if we want to disable user's logger for a feature, but still ensure that user code can run

    """

    def __init__(self) -> None:
        super().__init__()
        self._experiment = DummyExperiment()

    @property
    def experiment(self) -> DummyExperiment:
        """Return the experiment object associated with this logger."""
        return self._experiment

    @override
    def log_metrics(self, *args: Any, **kwargs: Any) -> None:
        pass

    @override
    def log_hyperparams(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    @override
    def name(self) -> str:
        """Return the experiment name."""
        return ''

    @property
    @override
    def version(self) -> str:
        """Return the experiment version."""
        return ''

    def __getitem__(self, idx: int) -> 'DummyLogger':
        return self

    def __getattr__(self, name: str) -> Callable:
        """Allows the DummyLogger to be called with arbitrary methods, to avoid AttributeErrors."""

        def method(*args: Any, **kwargs: Any) -> None:
            return None
        return method