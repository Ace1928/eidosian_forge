import torch
from torch.ao.quantization.observer import ObserverBase
def _calculate_min_max_stats(self, x_copy):
    """Calculates and stores the per_channel min, max stats with forward values.
        Does calculation based on channel axis: self.ch_axis

        Args
            x_copy: A copy of the forward data

        Returns the passed in x_copy
        """
    min_val = self.min_val
    max_val = self.max_val
    x_dim = x_copy.size()
    new_axis_list = [i for i in range(len(x_dim))]
    new_axis_list[self.ch_axis] = 0
    new_axis_list[0] = self.ch_axis
    y = x_copy.permute(new_axis_list)
    y = y.to(self.min_val.dtype)
    y = torch.flatten(y, start_dim=1)
    if min_val.numel() == 0 or max_val.numel() == 0:
        min_val, max_val = torch.aminmax(y, dim=1)
    else:
        min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
        min_val = torch.min(min_val_cur, min_val)
        max_val = torch.max(max_val_cur, max_val)
    self.min_val.resize_(min_val.shape)
    self.max_val.resize_(max_val.shape)
    self.min_val.copy_(min_val)
    self.max_val.copy_(max_val)
    return x_copy