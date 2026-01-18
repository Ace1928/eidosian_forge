import logging
import os
from typing import Any, Callable, Dict, Optional
from typing_extensions import override
from lightning_fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning_fabric.utilities.cloud_io import _atomic_save, get_filesystem
from lightning_fabric.utilities.cloud_io import _load as pl_load
from lightning_fabric.utilities.types import _PATH
class TorchCheckpointIO(CheckpointIO):
    """CheckpointIO that utilizes :func:`torch.save` and :func:`torch.load` to save and load checkpoints respectively,
    common for most use cases.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    """

    @override
    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any]=None) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            path: write-target path
            storage_options: not used in ``TorchCheckpointIO.save_checkpoint``

        Raises:
            TypeError:
                If ``storage_options`` arg is passed in

        """
        if storage_options is not None:
            raise TypeError(f"`Trainer.save_checkpoint(..., storage_options=...)` with `storage_options` arg is not supported for `{self.__class__.__name__}`. Please implement your custom `CheckpointIO` to define how you'd like to use `storage_options`.")
        fs = get_filesystem(path)
        fs.makedirs(os.path.dirname(path), exist_ok=True)
        _atomic_save(checkpoint, path)

    @override
    def load_checkpoint(self, path: _PATH, map_location: Optional[Callable]=lambda storage, loc: storage) -> Dict[str, Any]:
        """Loads checkpoint using :func:`torch.load`, with additional handling for ``fsspec`` remote loading of files.

        Args:
            path: Path to checkpoint
            map_location: a function, :class:`torch.device`, string or a dict specifying how to remap storage
                locations.

        Returns: The loaded checkpoint.

        Raises:
            FileNotFoundError: If ``path`` is not found by the ``fsspec`` filesystem

        """
        fs = get_filesystem(path)
        if not fs.exists(path):
            raise FileNotFoundError(f'Checkpoint file not found: {path}')
        return pl_load(path, map_location=map_location)

    @override
    def remove_checkpoint(self, path: _PATH) -> None:
        """Remove checkpoint file from the filesystem.

        Args:
            path: Path to checkpoint

        """
        fs = get_filesystem(path)
        if fs.exists(path):
            fs.rm(path, recursive=True)
            log.debug(f'Removed checkpoint: {path}')