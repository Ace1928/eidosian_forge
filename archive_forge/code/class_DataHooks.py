from typing import Any, Dict, Optional
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from pytorch_lightning.utilities import move_data_to_device
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
class DataHooks:
    """Hooks to be used for data related stuff."""

    def __init__(self) -> None:
        """
        Attributes:
            prepare_data_per_node:
                If True, each LOCAL_RANK=0 will call prepare data.
                Otherwise only NODE_RANK=0, LOCAL_RANK=0 will prepare data.
            allow_zero_length_dataloader_with_multiple_devices:
                If True, dataloader with zero length within local rank is allowed.
                Default value is False.
        """
        super().__init__()
        self.prepare_data_per_node: bool = True
        self.allow_zero_length_dataloader_with_multiple_devices: bool = False

    def prepare_data(self) -> None:
        """Use this to download and prepare data. Downloading and saving data with multiple processes (distributed
        settings) will result in corrupted data. Lightning ensures this method is called only within a single process,
        so you can safely add your downloading logic within.

        .. warning:: DO NOT set state to the model (use ``setup`` instead)
            since this is NOT called on every device

        Example::

            def prepare_data(self):
                # good
                download_data()
                tokenize()
                etc()

                # bad
                self.split = data_split
                self.some_state = some_other_state()

        In a distributed environment, ``prepare_data`` can be called in two ways
        (using :ref:`prepare_data_per_node<common/lightning_module:prepare_data_per_node>`)

        1. Once per node. This is the default and is only called on LOCAL_RANK=0.
        2. Once in total. Only called on GLOBAL_RANK=0.

        Example::

            # DEFAULT
            # called once per node on LOCAL_RANK=0 of that node
            class LitDataModule(LightningDataModule):
                def __init__(self):
                    super().__init__()
                    self.prepare_data_per_node = True


            # call on GLOBAL_RANK=0 (great for shared file systems)
            class LitDataModule(LightningDataModule):
                def __init__(self):
                    super().__init__()
                    self.prepare_data_per_node = False

        This is called before requesting the dataloaders:

        .. code-block:: python

            model.prepare_data()
            initialize_distributed()
            model.setup(stage)
            model.train_dataloader()
            model.val_dataloader()
            model.test_dataloader()
            model.predict_dataloader()

        """

    def setup(self, stage: str) -> None:
        """Called at the beginning of fit (train + validate), validate, test, or predict. This is a good hook when you
        need to build models dynamically or adjust something about them. This hook is called on every process when
        using DDP.

        Args:
            stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``

        Example::

            class LitModel(...):
                def __init__(self):
                    self.l1 = None

                def prepare_data(self):
                    download_data()
                    tokenize()

                    # don't do this
                    self.something = else

                def setup(self, stage):
                    data = load_data(...)
                    self.l1 = nn.Linear(28, data.num_classes)

        """

    def teardown(self, stage: str) -> None:
        """Called at the end of fit (train + validate), validate, test, or predict.

        Args:
            stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``

        """

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """An iterable or collection of iterables specifying training samples.

        For more information about multiple dataloaders, see this :ref:`section <multiple-dataloaders>`.

        The dataloader you return will not be reloaded unless you set
        :paramref:`~pytorch_lightning.trainer.trainer.Trainer.reload_dataloaders_every_n_epochs` to
        a positive integer.

        For data processing use the following pattern:

            - download in :meth:`prepare_data`
            - process and split in :meth:`setup`

        However, the above are only necessary for distributed processing.

        .. warning:: do not assign state in prepare_data

        - :meth:`~pytorch_lightning.trainer.trainer.Trainer.fit`
        - :meth:`prepare_data`
        - :meth:`setup`

        Note:
            Lightning tries to add the correct sampler for distributed and arbitrary hardware.
            There is no need to set it yourself.

        """
        raise MisconfigurationException('`train_dataloader` must be implemented to be used with the Lightning Trainer')

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """An iterable or collection of iterables specifying test samples.

        For more information about multiple dataloaders, see this :ref:`section <multiple-dataloaders>`.

        For data processing use the following pattern:

            - download in :meth:`prepare_data`
            - process and split in :meth:`setup`

        However, the above are only necessary for distributed processing.

        .. warning:: do not assign state in prepare_data


        - :meth:`~pytorch_lightning.trainer.trainer.Trainer.test`
        - :meth:`prepare_data`
        - :meth:`setup`

        Note:
            Lightning tries to add the correct sampler for distributed and arbitrary hardware.
            There is no need to set it yourself.

        Note:
            If you don't need a test dataset and a :meth:`test_step`, you don't need to implement
            this method.

        """
        raise MisconfigurationException('`test_dataloader` must be implemented to be used with the Lightning Trainer')

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """An iterable or collection of iterables specifying validation samples.

        For more information about multiple dataloaders, see this :ref:`section <multiple-dataloaders>`.

        The dataloader you return will not be reloaded unless you set
        :paramref:`~pytorch_lightning.trainer.trainer.Trainer.reload_dataloaders_every_n_epochs` to
        a positive integer.

        It's recommended that all data downloads and preparation happen in :meth:`prepare_data`.

        - :meth:`~pytorch_lightning.trainer.trainer.Trainer.fit`
        - :meth:`~pytorch_lightning.trainer.trainer.Trainer.validate`
        - :meth:`prepare_data`
        - :meth:`setup`

        Note:
            Lightning tries to add the correct sampler for distributed and arbitrary hardware
            There is no need to set it yourself.

        Note:
            If you don't need a validation dataset and a :meth:`validation_step`, you don't need to
            implement this method.

        """
        raise MisconfigurationException('`val_dataloader` must be implemented to be used with the Lightning Trainer')

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        """An iterable or collection of iterables specifying prediction samples.

        For more information about multiple dataloaders, see this :ref:`section <multiple-dataloaders>`.

        It's recommended that all data downloads and preparation happen in :meth:`prepare_data`.

        - :meth:`~pytorch_lightning.trainer.trainer.Trainer.predict`
        - :meth:`prepare_data`
        - :meth:`setup`

        Note:
            Lightning tries to add the correct sampler for distributed and arbitrary hardware
            There is no need to set it yourself.

        Return:
            A :class:`torch.utils.data.DataLoader` or a sequence of them specifying prediction samples.

        """
        raise MisconfigurationException('`predict_dataloader` must be implemented to be used with the Lightning Trainer')

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """Override this hook if your :class:`~torch.utils.data.DataLoader` returns tensors wrapped in a custom data
        structure.

        The data types listed below (and any arbitrary nesting of them) are supported out of the box:

        - :class:`torch.Tensor` or anything that implements `.to(...)`
        - :class:`list`
        - :class:`dict`
        - :class:`tuple`

        For anything else, you need to define how the data is moved to the target device (CPU, GPU, TPU, ...).

        Note:
            This hook should only transfer the data and not modify it, nor should it move the data to
            any other device than the one passed in as argument (unless you know what you are doing).
            To check the current state of execution of this hook you can use
            ``self.trainer.training/testing/validating/predicting`` so that you can
            add different logic as per your requirement.

        Args:
            batch: A batch of data that needs to be transferred to a new device.
            device: The target device as defined in PyTorch.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A reference to the data on the new device.

        Example::

            def transfer_batch_to_device(self, batch, device, dataloader_idx):
                if isinstance(batch, CustomBatch):
                    # move all tensors in your custom data structure to the device
                    batch.samples = batch.samples.to(device)
                    batch.targets = batch.targets.to(device)
                elif dataloader_idx == 0:
                    # skip device transfer for the first dataloader or anything you wish
                    pass
                else:
                    batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
                return batch

        Raises:
            MisconfigurationException:
                If using IPUs, ``Trainer(accelerator='ipu')``.

        See Also:
            - :meth:`move_data_to_device`
            - :meth:`apply_to_collection`

        """
        return move_data_to_device(batch, device)

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Override to alter or apply batch augmentations to your batch before it is transferred to the device.

        Note:
            To check the current state of execution of this hook you can use
            ``self.trainer.training/testing/validating/predicting`` so that you can
            add different logic as per your requirement.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data

        Example::

            def on_before_batch_transfer(self, batch, dataloader_idx):
                batch['x'] = transforms(batch['x'])
                return batch

        See Also:
            - :meth:`on_after_batch_transfer`
            - :meth:`transfer_batch_to_device`

        """
        return batch

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Override to alter or apply batch augmentations to your batch after it is transferred to the device.

        Note:
            To check the current state of execution of this hook you can use
            ``self.trainer.training/testing/validating/predicting`` so that you can
            add different logic as per your requirement.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data

        Example::

            def on_after_batch_transfer(self, batch, dataloader_idx):
                batch['x'] = gpu_transforms(batch['x'])
                return batch

        Raises:
            MisconfigurationException:
                If using IPUs, ``Trainer(accelerator='ipu')``.

        See Also:
            - :meth:`on_before_batch_transfer`
            - :meth:`transfer_batch_to_device`

        """
        return batch