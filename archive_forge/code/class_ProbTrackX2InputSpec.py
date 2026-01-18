import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class ProbTrackX2InputSpec(ProbTrackXBaseInputSpec):
    simple = traits.Bool(desc='rack from a list of voxels (seed must be a ASCII list of coordinates)', argstr='--simple')
    fopd = File(exists=True, desc='Other mask for binning tract distribution', argstr='--fopd=%s')
    waycond = traits.Enum('OR', 'AND', argstr='--waycond=%s', desc='Waypoint condition. Either "AND" (default) or "OR"')
    wayorder = traits.Bool(desc='Reject streamlines that do not hit waypoints in given order. Only valid if waycond=AND', argstr='--wayorder')
    onewaycondition = traits.Bool(desc='Apply waypoint conditions to each half tract separately', argstr='--onewaycondition')
    omatrix1 = traits.Bool(desc='Output matrix1 - SeedToSeed Connectivity', argstr='--omatrix1')
    distthresh1 = traits.Float(argstr='--distthresh1=%.3f', desc='Discards samples (in matrix1) shorter than this threshold (in mm - default=0)')
    omatrix2 = traits.Bool(desc='Output matrix2 - SeedToLowResMask', argstr='--omatrix2', requires=['target2'])
    target2 = File(exists=True, desc='Low resolution binary brain mask for storing connectivity distribution in matrix2 mode', argstr='--target2=%s')
    omatrix3 = traits.Bool(desc='Output matrix3 (NxN connectivity matrix)', argstr='--omatrix3', requires=['target3', 'lrtarget3'])
    target3 = File(exists=True, desc='Mask used for NxN connectivity matrix (or Nxn if lrtarget3 is set)', argstr='--target3=%s')
    lrtarget3 = File(exists=True, desc='Column-space mask used for Nxn connectivity matrix', argstr='--lrtarget3=%s')
    distthresh3 = traits.Float(argstr='--distthresh3=%.3f', desc='Discards samples (in matrix3) shorter than this threshold (in mm - default=0)')
    omatrix4 = traits.Bool(desc='Output matrix4 - DtiMaskToSeed (special Oxford Sparse Format)', argstr='--omatrix4')
    colmask4 = File(exists=True, desc='Mask for columns of matrix4 (default=seed mask)', argstr='--colmask4=%s')
    target4 = File(exists=True, desc='Brain mask in DTI space', argstr='--target4=%s')
    meshspace = traits.Enum('caret', 'freesurfer', 'first', 'vox', argstr='--meshspace=%s', desc='Mesh reference space - either "caret" (default) or "freesurfer" or "first" or "vox"')