import os
from ..base import TraitedSpec, File, traits, isdefined
from .base import get_custom_path, NiftyRegCommand, NiftyRegCommandInputSpec
from ...utils.filemanip import split_filename
def _find_input(self):
    inputs = [self.inputs.def_input, self.inputs.disp_input, self.inputs.flow_input, self.inputs.comp_input, self.inputs.comp_input2, self.inputs.upd_s_form_input, self.inputs.inv_aff_input, self.inputs.inv_nrr_input, self.inputs.half_input, self.inputs.make_aff_input, self.inputs.aff_2_rig_input, self.inputs.flirt_2_nr_input]
    entries = []
    for entry in inputs:
        if isdefined(entry):
            entries.append(entry)
            _, _, ext = split_filename(entry)
            if ext == '.nii' or ext == '.nii.gz' or ext == '.hdr':
                return entry
    if len(entries):
        return entries[0]
    return None