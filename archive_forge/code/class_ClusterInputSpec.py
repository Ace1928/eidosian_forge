import os
from glob import glob
from shutil import rmtree
from string import Template
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import simplify_list, ensure_list
from ...utils.misc import human_order_sorted
from ...external.due import BibTeX
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class ClusterInputSpec(FSLCommandInputSpec):
    in_file = File(argstr='--in=%s', mandatory=True, exists=True, desc='input volume')
    threshold = traits.Float(argstr='--thresh=%.10f', mandatory=True, desc='threshold for input volume')
    out_index_file = traits.Either(traits.Bool, File, argstr='--oindex=%s', desc='output of cluster index (in size order)', hash_files=False)
    out_threshold_file = traits.Either(traits.Bool, File, argstr='--othresh=%s', desc='thresholded image', hash_files=False)
    out_localmax_txt_file = traits.Either(traits.Bool, File, argstr='--olmax=%s', desc='local maxima text file', hash_files=False)
    out_localmax_vol_file = traits.Either(traits.Bool, File, argstr='--olmaxim=%s', desc='output of local maxima volume', hash_files=False)
    out_size_file = traits.Either(traits.Bool, File, argstr='--osize=%s', desc='filename for output of size image', hash_files=False)
    out_max_file = traits.Either(traits.Bool, File, argstr='--omax=%s', desc='filename for output of max image', hash_files=False)
    out_mean_file = traits.Either(traits.Bool, File, argstr='--omean=%s', desc='filename for output of mean image', hash_files=False)
    out_pval_file = traits.Either(traits.Bool, File, argstr='--opvals=%s', desc='filename for image output of log pvals', hash_files=False)
    pthreshold = traits.Float(argstr='--pthresh=%.10f', requires=['dlh', 'volume'], desc='p-threshold for clusters')
    peak_distance = traits.Float(argstr='--peakdist=%.10f', desc='minimum distance between local maxima/minima, in mm (default 0)')
    cope_file = File(argstr='--cope=%s', desc='cope volume')
    volume = traits.Int(argstr='--volume=%d', desc='number of voxels in the mask')
    dlh = traits.Float(argstr='--dlh=%.10f', desc='smoothness estimate = sqrt(det(Lambda))')
    fractional = traits.Bool(False, usedefault=True, argstr='--fractional', desc='interprets the threshold as a fraction of the robust range')
    connectivity = traits.Int(argstr='--connectivity=%d', desc='the connectivity of voxels (default 26)')
    use_mm = traits.Bool(False, usedefault=True, argstr='--mm', desc='use mm, not voxel, coordinates')
    find_min = traits.Bool(False, usedefault=True, argstr='--min', desc='find minima instead of maxima')
    no_table = traits.Bool(False, usedefault=True, argstr='--no_table', desc='suppresses printing of the table info')
    minclustersize = traits.Bool(False, usedefault=True, argstr='--minclustersize', desc='prints out minimum significant cluster size')
    xfm_file = File(argstr='--xfm=%s', desc='filename for Linear: input->standard-space transform. Non-linear: input->highres transform')
    std_space_file = File(argstr='--stdvol=%s', desc='filename for standard-space volume')
    num_maxima = traits.Int(argstr='--num=%d', desc='no of local maxima to report')
    warpfield_file = File(argstr='--warpvol=%s', desc='file contining warpfield')