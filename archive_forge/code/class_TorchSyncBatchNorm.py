from abc import ABC, abstractmethod
import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import override
class TorchSyncBatchNorm(LayerSync):
    """A plugin that wraps all batch normalization layers of a model with synchronization logic for multiprocessing.

    This plugin has no effect in single-device operation.

    """

    @override
    def apply(self, model: Module) -> Module:
        """Add global batchnorm for a model spread across multiple GPUs and nodes.

        Override this method to synchronize batchnorm layers between specific process groups instead
        of the whole world.

        Args:
            model: Reference to the current LightningModule

        Return:
            LightningModule with batchnorm layers synchronized within the process groups.

        """
        return torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    @override
    def revert(self, model: Module) -> Module:
        """Convert the wrapped batchnorm layers back to regular batchnorm layers.

        Args:
            model: Reference to the current LightningModule

        Return:
            LightningModule with regular batchnorm layers that will no longer sync across processes.

        """
        converted_module = model
        if isinstance(model, torch.nn.modules.batchnorm.SyncBatchNorm):
            converted_module = _BatchNormXd(model.num_features, model.eps, model.momentum, model.affine, model.track_running_stats)
            if model.affine:
                with torch.no_grad():
                    converted_module.weight = model.weight
                    converted_module.bias = model.bias
            converted_module.running_mean = model.running_mean
            converted_module.running_var = model.running_var
            converted_module.num_batches_tracked = model.num_batches_tracked
            if hasattr(model, 'qconfig'):
                converted_module.qconfig = model.qconfig
        for name, child in model.named_children():
            converted_module.add_module(name, self.revert(child))
        del model
        return converted_module