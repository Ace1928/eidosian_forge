import sys
from typing import Any
import pytorch_lightning as pl
class HPUParallelStrategy:

    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError('The `HPUParallelStrategy` class has been moved to an external package. Install the extension package as `pip install lightning-habana` and import with `from lightning_habana import HPUParallelStrategy`. Please see: https://github.com/Lightning-AI/lightning-Habana for more details.')

    def setup(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError

    def get_device_stats(self, *_: Any, **__: Any) -> dict:
        raise NotImplementedError