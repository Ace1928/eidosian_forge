import torch
from accelerate import Accelerator, DistributedType
def _sync_and_avg_model_params(self):
    """
        Synchronize + Average model parameters across all GPUs
        """
    self.accelerator.wait_for_everyone()
    with self.accelerator.autocast():
        for param in self.model.parameters():
            param.data = self.accelerator.reduce(param.data, reduction='mean')