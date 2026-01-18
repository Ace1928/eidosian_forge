from ..base import TraitedSpec, File, traits, CommandLineInputSpec
from .base import WBCommand
from ... import logging

    Smooth a CIFTI file

    The input cifti file must have a brain models mapping on the chosen
    dimension, columns for .dtseries, and either for .dconn.  By default,
    data in different structures is smoothed independently (i.e., "parcel
    constrained" smoothing), so volume structures that touch do not smooth
    across this boundary.  Specify ``merged_volume`` to ignore these
    boundaries. Surface smoothing uses the ``GEO_GAUSS_AREA`` smoothing method.

    The ``*_corrected_areas`` options are intended for when it is unavoidable
    to smooth on group average surfaces, it is only an approximate correction
    for the reduction of structure in a group average surface.  It is better
    to smooth the data on individuals before averaging, when feasible.

    The ``fix_zeros_*`` options will treat values of zero as lack of data, and
    not use that value when generating the smoothed values, but will fill
    zeros with extrapolated values.  The ROI should have a brain models
    mapping along columns, exactly matching the mapping of the chosen
    direction in the input file.  Data outside the ROI is ignored.

    >>> from nipype.interfaces.workbench import CiftiSmooth
    >>> smooth = CiftiSmooth()
    >>> smooth.inputs.in_file = 'sub-01_task-rest.dtseries.nii'
    >>> smooth.inputs.sigma_surf = 4
    >>> smooth.inputs.sigma_vol = 4
    >>> smooth.inputs.direction = 'COLUMN'
    >>> smooth.inputs.right_surf = 'sub-01.R.midthickness.32k_fs_LR.surf.gii'
    >>> smooth.inputs.left_surf = 'sub-01.L.midthickness.32k_fs_LR.surf.gii'
    >>> smooth.cmdline
    'wb_command -cifti-smoothing sub-01_task-rest.dtseries.nii 4.0 4.0 COLUMN     smoothed_sub-01_task-rest.dtseries.nii     -left-surface sub-01.L.midthickness.32k_fs_LR.surf.gii     -right-surface sub-01.R.midthickness.32k_fs_LR.surf.gii'
    