from ..base import File, TraitedSpec, traits, isdefined, CommandLineInputSpec
from .base import NiftyFitCommand
from ..niftyreg.base import get_custom_path
Interface for executable dwi_tool from Niftyfit platform.

    Use DwiTool.

    Diffusion-Weighted MR Prediction.
    Predicts DWI from previously fitted models and calculates model derived
    maps.

    `Source code <https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyFit-Release>`_

    Examples
    --------

    >>> from nipype.interfaces import niftyfit
    >>> dwi_tool = niftyfit.DwiTool(dti_flag=True)
    >>> dwi_tool.inputs.source_file = 'dwi.nii.gz'
    >>> dwi_tool.inputs.bvec_file = 'bvecs'
    >>> dwi_tool.inputs.bval_file = 'bvals'
    >>> dwi_tool.inputs.mask_file = 'mask.nii.gz'
    >>> dwi_tool.inputs.b0_file = 'b0.nii.gz'
    >>> dwi_tool.inputs.rgbmap_file = 'rgb_map.nii.gz'
    >>> dwi_tool.cmdline
    'dwi_tool -source dwi.nii.gz -bval bvals -bvec bvecs -b0 b0.nii.gz -mask mask.nii.gz -dti -famap dwi_famap.nii.gz -logdti2 dwi_logdti2.nii.gz -mcmap dwi_mcmap.nii.gz -mdmap dwi_mdmap.nii.gz -rgbmap rgb_map.nii.gz -syn dwi_syn.nii.gz -v1map dwi_v1map.nii.gz'

    