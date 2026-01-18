import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class AlignEpiAnatPyOutputSpec(TraitedSpec):
    anat_al_orig = File(desc='A version of the anatomy that is aligned to the EPI')
    epi_al_orig = File(desc='A version of the EPI dataset aligned to the anatomy')
    epi_tlrc_al = File(desc='A version of the EPI dataset aligned to a standard template')
    anat_al_mat = File(desc='matrix to align anatomy to the EPI')
    epi_al_mat = File(desc='matrix to align EPI to anatomy')
    epi_vr_al_mat = File(desc='matrix to volume register EPI')
    epi_reg_al_mat = File(desc='matrix to volume register and align epi to anatomy')
    epi_al_tlrc_mat = File(desc='matrix to volume register and align epito anatomy and put into standard space')
    epi_vr_motion = File(desc='motion parameters from EPI time-seriesregistration (tsh included in name if slicetiming correction is also included).')
    skullstrip = File(desc='skull-stripped (not aligned) volume')