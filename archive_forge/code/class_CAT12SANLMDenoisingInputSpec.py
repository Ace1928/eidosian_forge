import os
from pathlib import Path
from nipype.interfaces.base import (
from nipype.interfaces.cat12.base import Cell
from nipype.interfaces.spm import SPMCommand
from nipype.interfaces.spm.base import (
from nipype.utils.filemanip import split_filename, fname_presuffix
class CAT12SANLMDenoisingInputSpec(SPMCommandInputSpec):
    in_files = InputMultiPath(ImageFileSPM(exists=True), field='data', desc='Images for filtering.', mandatory=True, copyfile=False)
    spm_type = traits.Enum('float32', 'uint16', 'uint8', 'same', field='spm_type', usedefault=True, desc="Data type of the output images. 'same' matches the input image type.")
    intlim = traits.Int(field='intlim', default_value=100, usedefault=True, desc='intensity limitation (default = 100)')
    filename_prefix = traits.Str(field='prefix', default_value='sanlm_', usedefault=True, desc='Filename prefix. Specify  the  string  to be prepended to the filenames of the filtered image file(s).')
    filename_suffix = traits.Str(field='suffix', default_value='', usedefault=True, desc='Filename suffix. Specify  the  string  to  be  appended  to the filenames of the filtered image file(s).')
    addnoise = traits.Float(default_value=0.5, usedefault=True, field='addnoise', desc='Strength of additional noise in noise-free regions.\n        Add  minimal  amount  of noise in regions without any noise to avoid image segmentation problems.\n        This parameter defines the strength of additional noise as percentage of the average signal intensity.')
    rician = traits.Bool(True, field='rician', usedefault=True, desc='Rician noise\n        MRIs  can  have  Gaussian  or  Rician  distributed  noise with uniform or nonuniform variance across the image.\n        If SNR is high enough (>3)  noise  can  be  well  approximated by Gaussian noise in the foreground. However, for\n        SENSE reconstruction or DTI data a Rician distribution is expected. Please note that the Rician noise estimation\n        is sensitive for large signals in the neighbourhood and can lead to artefacts, e.g. cortex can be affected by\n        very high values in the scalp or in blood vessels.')
    replace_nan_and_inf = traits.Bool(True, field='replaceNANandINF', usedefault=True, desc='Replace NAN by 0, -INF by the minimum and INF by the maximum of the image.')
    noisecorr_strength = traits.Enum('-Inf', 2, 4, field='nlmfilter.optimized.NCstr', usedefault=True, desc='Strength of Noise Corrections\n        Strength  of  the  (sub-resolution)  spatial  adaptive    non local means (SANLM) noise correction. Please note\n        that the filter strength is automatically  estimated.  Change this parameter only for specific conditions. The\n        "light" option applies half of the filter strength of the adaptive  "medium"  cases,  whereas  the  "strong"\n        option  uses  the  full  filter  strength,  force sub-resolution filtering and applies an additional  iteration.\n        Sub-resolution  filtering  is  only  used  in  case  of  high image resolution below 0.8 mm or in case of the\n        "strong" option. light = 2, medium = -Inf, strong = 4')