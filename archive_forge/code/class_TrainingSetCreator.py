from ..base import (
import os
class TrainingSetCreator(BaseInterface):
    """Goes through set of provided melodic output directories, to find all
    the ones that have a hand_labels_noise.txt file in them.

    This is outsourced as a separate class, so that the pipeline is
    rerun every time a handlabeled file has been changed, or a new one
    created.

    """
    input_spec = TrainingSetCreatorInputSpec
    output_spec = TrainingSetCreatorOutputSpec
    _always_run = True

    def _run_interface(self, runtime):
        mel_icas = []
        for item in self.inputs.mel_icas_in:
            if os.path.exists(os.path.join(item, 'hand_labels_noise.txt')):
                mel_icas.append(item)
        if len(mel_icas) == 0:
            raise Exception('%s did not find any hand_labels_noise.txt files in the following directories: %s' % (self.__class__.__name__, mel_icas))
        return runtime

    def _list_outputs(self):
        mel_icas = []
        for item in self.inputs.mel_icas_in:
            if os.path.exists(os.path.join(item, 'hand_labels_noise.txt')):
                mel_icas.append(item)
        outputs = self._outputs().get()
        outputs['mel_icas_out'] = mel_icas
        return outputs