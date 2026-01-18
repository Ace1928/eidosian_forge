import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class BinarizeInputSpec(FSTraitedSpec):
    in_file = File(exists=True, argstr='--i %s', mandatory=True, copyfile=False, desc='input volume')
    min = traits.Float(argstr='--min %f', xor=['wm_ven_csf'], desc='min thresh')
    max = traits.Float(argstr='--max %f', xor=['wm_ven_csf'], desc='max thresh')
    rmin = traits.Float(argstr='--rmin %f', desc='compute min based on rmin*globalmean')
    rmax = traits.Float(argstr='--rmax %f', desc='compute max based on rmax*globalmean')
    match = traits.List(traits.Int, argstr='--match %d...', desc='match instead of threshold')
    wm = traits.Bool(argstr='--wm', desc='set match vals to 2 and 41 (aseg for cerebral WM)')
    ventricles = traits.Bool(argstr='--ventricles', desc='set match vals those for aseg ventricles+choroid (not 4th)')
    wm_ven_csf = traits.Bool(argstr='--wm+vcsf', xor=['min', 'max'], desc='WM and ventricular CSF, including choroid (not 4th)')
    binary_file = File(argstr='--o %s', genfile=True, desc='binary output volume')
    out_type = traits.Enum('nii', 'nii.gz', 'mgz', argstr='', desc='output file type')
    count_file = traits.Either(traits.Bool, File, argstr='--count %s', desc='save number of hits in ascii file (hits, ntotvox, pct)')
    bin_val = traits.Int(argstr='--binval %d', desc='set vox within thresh to val (default is 1)')
    bin_val_not = traits.Int(argstr='--binvalnot %d', desc='set vox outside range to val (default is 0)')
    invert = traits.Bool(argstr='--inv', desc='set binval=0, binvalnot=1')
    frame_no = traits.Int(argstr='--frame %s', desc='use 0-based frame of input (default is 0)')
    merge_file = File(exists=True, argstr='--merge %s', desc='merge with mergevol')
    mask_file = File(exists=True, argstr='--mask maskvol', desc='must be within mask')
    mask_thresh = traits.Float(argstr='--mask-thresh %f', desc='set thresh for mask')
    abs = traits.Bool(argstr='--abs', desc='take abs of invol first (ie, make unsigned)')
    bin_col_num = traits.Bool(argstr='--bincol', desc='set binarized voxel value to its column number')
    zero_edges = traits.Bool(argstr='--zero-edges', desc='zero the edge voxels')
    zero_slice_edge = traits.Bool(argstr='--zero-slice-edges', desc='zero the edge slice voxels')
    dilate = traits.Int(argstr='--dilate %d', desc='niters: dilate binarization in 3D')
    erode = traits.Int(argstr='--erode  %d', desc='nerode: erode binarization in 3D (after any dilation)')
    erode2d = traits.Int(argstr='--erode2d %d', desc='nerode2d: erode binarization in 2D (after any 3D erosion)')