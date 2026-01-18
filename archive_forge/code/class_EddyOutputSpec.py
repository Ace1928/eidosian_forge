import os
import numpy as np
import nibabel as nb
import warnings
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import traits, TraitedSpec, InputMultiPath, File, isdefined
from .base import FSLCommand, FSLCommandInputSpec, Info
class EddyOutputSpec(TraitedSpec):
    out_corrected = File(exists=True, desc='4D image file containing all the corrected volumes')
    out_parameter = File(exists=True, desc='Text file with parameters defining the field and movement for each scan')
    out_rotated_bvecs = File(exists=True, desc='File containing rotated b-values for all volumes')
    out_movement_rms = File(exists=True, desc="Summary of the 'total movement' in each volume")
    out_restricted_movement_rms = File(exists=True, desc="Summary of the 'total movement' in each volume disregarding translation in the PE direction")
    out_shell_alignment_parameters = File(exists=True, desc='Text file containing rigid body movement parameters between the different shells as estimated by a post-hoc mutual information based registration')
    out_shell_pe_translation_parameters = File(exists=True, desc='Text file containing translation along the PE-direction between the different shells as estimated by a post-hoc mutual information based registration')
    out_shell_pe_translation_parameters = File(exists=True, desc='Text file containing translation along the PE-direction between the different shells as estimated by a post-hoc mutual information based registration')
    out_outlier_map = File(exists=True, desc='Matrix where rows represent volumes and columns represent slices. "0" indicates that scan-slice is not an outlier and "1" indicates that it is')
    out_outlier_n_stdev_map = File(exists=True, desc='Matrix where rows represent volumes and columns represent slices. Values indicate number of standard deviations off the mean difference between observation and prediction is')
    out_outlier_n_sqr_stdev_map = File(exists=True, desc='Matrix where rows represent volumes and columns represent slices. Values indicate number of standard deivations off the square root of the mean squared difference between observation and prediction is')
    out_outlier_report = File(exists=True, desc='Text file with a plain language report on what outlier slices eddy has found')
    out_outlier_free = File(exists=True, desc='4D image file not corrected for susceptibility or eddy-current distortions or subject movement but with outlier slices replaced')
    out_movement_over_time = File(exists=True, desc='Text file containing translations (mm) and rotations (radians) for each excitation')
    out_cnr_maps = File(exists=True, desc='path/name of file with the cnr_maps')
    out_residuals = File(exists=True, desc='path/name of file with the residuals')