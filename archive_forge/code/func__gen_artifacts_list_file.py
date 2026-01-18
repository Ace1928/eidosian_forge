from ..base import (
import os
def _gen_artifacts_list_file(self, mel_ica, thresh):
    _, trained_wts_file = os.path.split(self.inputs.trained_wts_file)
    trained_wts_filestem = trained_wts_file.split('.')[0]
    filestem = 'fix4melview_' + trained_wts_filestem + '_thr'
    fname = os.path.join(mel_ica, filestem + str(thresh) + '.txt')
    return fname