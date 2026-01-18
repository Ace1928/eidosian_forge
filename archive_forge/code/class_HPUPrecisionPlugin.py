import sys
from typing import Any
import pytorch_lightning as pl
class HPUPrecisionPlugin:

    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError('The `HPUPrecisionPlugin` class has been moved to an external package. Install the extension package as `pip install lightning-habana` and import with `from lightning_habana import HPUPrecisionPlugin`. Please see: https://github.com/Lightning-AI/lightning-Habana for more details.')