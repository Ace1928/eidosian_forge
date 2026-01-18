import torch
from torch.ao.quantization.observer import ObserverBase
def _calculate_percentile_stats(self, x_copy):
    """Calculates and stores the per_channel percentile stats with forward values.
        Does calculation based on channel axis: self.ch_axis

        Args
            x_copy: A copy of the forward data

        Returns the passed in x_copy
        """
    x_dim = x_copy.size()
    new_axis_list = [i for i in range(len(x_dim))]
    new_axis_list[self.ch_axis] = 0
    new_axis_list[0] = self.ch_axis
    y = x_copy.permute(new_axis_list)
    y = y.to(self.min_val.dtype)
    y = torch.flatten(y, start_dim=1)
    y = y.to(dtype=self.min_val.dtype, device='cpu')
    quantiles_list = [0, self.comp_percentile, 1.0]
    quantiles_to_find = torch.tensor(quantiles_list, dtype=self.min_val.dtype)
    desired_quantiles = torch.quantile(y, quantiles_to_find, dim=self.ch_axis, interpolation='lower')
    zero_quantile = desired_quantiles[0]
    comp_quantile = desired_quantiles[1]
    hundreth_quartile = desired_quantiles[2]
    any_non_zero_quantile_value: torch.Tensor = (comp_quantile != torch.tensor([0])) | (hundreth_quartile != torch.tensor([0]))
    any_non_zero_quantile_value = any_non_zero_quantile_value.int()
    any_constant_channels: torch.Tensor = hundreth_quartile - zero_quantile == torch.tensor([0])
    any_constant_channels = any_constant_channels.int()
    quantile_ratios = hundreth_quartile / comp_quantile
    quantile_ratios = torch.nan_to_num(quantile_ratios)
    ratio_if_not_zero = any_non_zero_quantile_value * quantile_ratios
    if self.percentile_batches_tracked.shape[0] == 0 or self.average_percentile_ratio.shape[0] == 0:
        self.percentile_batches_tracked = torch.zeros_like(any_non_zero_quantile_value)
        self.average_percentile_ratio = torch.zeros_like(ratio_if_not_zero)
    if self.constant_channels.shape[0] == 0:
        self.constant_channels = torch.zeros_like(any_constant_channels)
    num_batches = self.percentile_batches_tracked
    average_ratio = self.average_percentile_ratio
    new_number_of_batches: torch.Tensor = num_batches + any_non_zero_quantile_value
    new_ratios: torch.Tensor = (average_ratio * num_batches + ratio_if_not_zero) / new_number_of_batches
    new_ratios = torch.nan_to_num(new_ratios)
    new_constant_count: torch.Tensor = self.constant_channels + any_constant_channels
    self.percentile_batches_tracked.copy_(new_number_of_batches)
    self.average_percentile_ratio.copy_(new_ratios)
    self.constant_channels.copy_(new_constant_count)
    return x_copy