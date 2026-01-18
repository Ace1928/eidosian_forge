import os
from typing import Any
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.callbacks import Checkpoint
class OnExceptionCheckpoint(Checkpoint):
    """Used to save a checkpoint on exception.

    Args:
        dirpath: directory to save the checkpoint file.
        filename: checkpoint filename. This must not include the extension.

    Raises:
        ValueError:
            If ``filename`` is empty.


    Example:
        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import OnExceptionCheckpoint
        >>> trainer = Trainer(callbacks=[OnExceptionCheckpoint(".")])

    """
    FILE_EXTENSION = '.ckpt'

    def __init__(self, dirpath: _PATH, filename: str='on_exception') -> None:
        super().__init__()
        if not filename:
            raise ValueError('The filename cannot be empty')
        self.dirpath = dirpath
        self.filename = filename

    @property
    def ckpt_path(self) -> str:
        return os.path.join(self.dirpath, self.filename + self.FILE_EXTENSION)

    @override
    def on_exception(self, trainer: 'pl.Trainer', *_: Any, **__: Any) -> None:
        trainer.save_checkpoint(self.ckpt_path)

    @override
    def teardown(self, trainer: 'pl.Trainer', *_: Any, **__: Any) -> None:
        trainer.strategy.remove_checkpoint(self.ckpt_path)