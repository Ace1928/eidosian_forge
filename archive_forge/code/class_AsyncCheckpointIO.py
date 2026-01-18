from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional
from typing_extensions import override
from lightning_fabric.plugins import CheckpointIO
from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO
class AsyncCheckpointIO(_WrappingCheckpointIO):
    """``AsyncCheckpointIO`` enables saving the checkpoints asynchronously in a thread.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Args:
        checkpoint_io: A checkpoint IO plugin that is used as the basis for async checkpointing.

    """

    def __init__(self, checkpoint_io: Optional['CheckpointIO']=None) -> None:
        super().__init__(checkpoint_io)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._error: Optional[BaseException] = None

    @override
    def save_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        """Uses the ``ThreadPoolExecutor`` to save the checkpoints using the base ``checkpoint_io``."""

        def _save_checkpoint(*args: Any, **kwargs: Any) -> None:
            try:
                assert self.checkpoint_io is not None
                self.checkpoint_io.save_checkpoint(*args, **kwargs)
            except BaseException as ex:
                self._error = ex
        self._executor.submit(_save_checkpoint, *args, **kwargs)
        if self._error:
            raise self._error

    @override
    def teardown(self) -> None:
        """This method is called to close the threads."""
        self._executor.shutdown(wait=True)
        if self._error:
            raise self._error