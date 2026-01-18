import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
def _format_metric(self, index):
    """
        Format the antsRegistration -m metric argument(s).

        Parameters
        ----------
        index: the stage index
        """
    name_input = self.inputs.metric[index]
    stage_inputs = dict(fixed_image=self.inputs.fixed_image[0], moving_image=self.inputs.moving_image[0], metric=name_input, weight=self.inputs.metric_weight[index], radius_or_bins=self.inputs.radius_or_number_of_bins[index], optional=self.inputs.radius_or_number_of_bins[index])
    if isdefined(self.inputs.sampling_strategy) and self.inputs.sampling_strategy:
        sampling_strategy = self.inputs.sampling_strategy[index]
        if sampling_strategy:
            stage_inputs['sampling_strategy'] = sampling_strategy
    if isdefined(self.inputs.sampling_percentage) and self.inputs.sampling_percentage:
        sampling_percentage = self.inputs.sampling_percentage[index]
        if sampling_percentage:
            stage_inputs['sampling_percentage'] = sampling_percentage
    if isinstance(name_input, list):
        items = list(stage_inputs.items())
        indexes = list(range(0, len(name_input)))
        specs = list()
        for i in indexes:
            temp = dict([(k, v[i]) for k, v in items])
            if len(self.inputs.fixed_image) == 1:
                temp['fixed_image'] = self.inputs.fixed_image[0]
            else:
                temp['fixed_image'] = self.inputs.fixed_image[i]
            if len(self.inputs.moving_image) == 1:
                temp['moving_image'] = self.inputs.moving_image[0]
            else:
                temp['moving_image'] = self.inputs.moving_image[i]
            specs.append(temp)
    else:
        specs = [stage_inputs]
    return [self._format_metric_argument(**spec) for spec in specs]