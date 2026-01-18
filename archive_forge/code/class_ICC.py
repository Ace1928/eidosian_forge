import os
from functools import lru_cache
import numpy as np
from numpy import ones, kron, mean, eye, hstack, tile
from numpy.linalg import pinv
import nibabel as nb
from ..interfaces.base import (
class ICC(BaseInterface):
    """
    Calculates Interclass Correlation Coefficient (3,1) as defined in
    P. E. Shrout & Joseph L. Fleiss (1979). "Intraclass Correlations: Uses in
    Assessing Rater Reliability". Psychological Bulletin 86 (2): 420-428. This
    particular implementation is aimed at relaibility (test-retest) studies.
    """
    input_spec = ICCInputSpec
    output_spec = ICCOutputSpec

    def _run_interface(self, runtime):
        maskdata = nb.load(self.inputs.mask).get_fdata()
        maskdata = np.logical_not(np.logical_or(maskdata == 0, np.isnan(maskdata)))
        session_datas = [[nb.load(fname).get_fdata()[maskdata].reshape(-1, 1) for fname in sessions] for sessions in self.inputs.subjects_sessions]
        list_of_sessions = [np.dstack(session_data) for session_data in session_datas]
        all_data = np.hstack(list_of_sessions)
        icc = np.zeros(session_datas[0][0].shape)
        session_F = np.zeros(session_datas[0][0].shape)
        session_var = np.zeros(session_datas[0][0].shape)
        subject_var = np.zeros(session_datas[0][0].shape)
        for x in range(icc.shape[0]):
            Y = all_data[x, :, :]
            icc[x], subject_var[x], session_var[x], session_F[x], _, _ = ICC_rep_anova(Y)
        nim = nb.load(self.inputs.subjects_sessions[0][0])
        new_data = np.zeros(nim.shape)
        new_data[maskdata] = icc.reshape(-1)
        new_img = nb.Nifti1Image(new_data, nim.affine, nim.header)
        nb.save(new_img, 'icc_map.nii')
        new_data = np.zeros(nim.shape)
        new_data[maskdata] = session_var.reshape(-1)
        new_img = nb.Nifti1Image(new_data, nim.affine, nim.header)
        nb.save(new_img, 'session_var_map.nii')
        new_data = np.zeros(nim.shape)
        new_data[maskdata] = subject_var.reshape(-1)
        new_img = nb.Nifti1Image(new_data, nim.affine, nim.header)
        nb.save(new_img, 'subject_var_map.nii')
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['icc_map'] = os.path.abspath('icc_map.nii')
        outputs['session_var_map'] = os.path.abspath('session_var_map.nii')
        outputs['subject_var_map'] = os.path.abspath('subject_var_map.nii')
        return outputs