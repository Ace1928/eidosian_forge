from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional
from typing_extensions import override
from lightning_fabric.plugins import CheckpointIO
from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO
This method is called to close the threads.