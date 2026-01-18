import sys
from typing import Any
import pytorch_lightning as pl
class IPUAccelerator:

    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError('The `IPUAccelerator` class has been moved to an external package. Install the extension package as `pip install lightning-graphcore` and import with `from lightning_graphcore import IPUAccelerator`. Please see: https://github.com/Lightning-AI/lightning-Graphcore for more details.')