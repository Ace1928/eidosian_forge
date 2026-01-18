from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from lightning_fabric.utilities.types import _PATH
class CheckpointIO(ABC):
    """Interface to save/load checkpoints as they are saved through the ``Strategy``.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Typically most plugins either use the Torch based IO Plugin; ``TorchCheckpointIO`` but may
    require particular handling depending on the plugin.

    In addition, you can pass a custom ``CheckpointIO`` by extending this class and passing it
    to the Trainer, i.e ``Trainer(plugins=[MyCustomCheckpointIO()])``.

    .. note::

        For some plugins, it is not possible to use a custom checkpoint plugin as checkpointing logic is not
        modifiable.

    """

    @abstractmethod
    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any]=None) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            path: write-target path
            storage_options: Optional parameters when saving the model/training states.

        """

    @abstractmethod
    def load_checkpoint(self, path: _PATH, map_location: Optional[Any]=None) -> Dict[str, Any]:
        """Load checkpoint from a path when resuming or loading ckpt for test/validate/predict stages.

        Args:
            path: Path to checkpoint
            map_location: a function, :class:`torch.device`, string or a dict specifying how to remap storage
                locations.

        Returns: The loaded checkpoint.

        """

    @abstractmethod
    def remove_checkpoint(self, path: _PATH) -> None:
        """Remove checkpoint file from the filesystem.

        Args:
            path: Path to checkpoint

        """

    def teardown(self) -> None:
        """This method is called to teardown the process."""