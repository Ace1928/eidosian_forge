from ..base import (
import os
class Training(CommandLine):
    """
    Train the classifier based on your own FEAT/MELODIC output directory.
    """
    input_spec = TrainingInputSpec
    output_spec = TrainingOutputSpec
    cmd = 'fix -t'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.trained_wts_filestem):
            outputs['trained_wts_file'] = os.path.abspath(self.inputs.trained_wts_filestem + '.RData')
        else:
            outputs['trained_wts_file'] = os.path.abspath('trained_wts_file.RData')
        return outputs