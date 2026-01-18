import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class ReHoInputSpec(CommandLineInputSpec):
    in_file = File(desc='input dataset', argstr='-inset %s', position=1, mandatory=True, exists=True)
    out_file = File(desc='Output dataset.', argstr='-prefix %s', name_source='in_file', name_template='%s_reho', keep_extension=True, position=0)
    chi_sq = traits.Bool(argstr='-chi_sq', desc="Output the Friedman chi-squared value in addition to the Kendall's W. This option is currently compatible only with the AFNI (BRIK/HEAD) output type; the chi-squared value will be the second sub-brick of the output dataset.")
    mask_file = File(desc='Mask within which ReHo should be calculated voxelwise', argstr='-mask %s')
    neighborhood = traits.Enum('faces', 'edges', 'vertices', xor=['sphere', 'ellipsoid'], argstr='-nneigh %s', desc='\nvoxels in neighborhood. can be:\n``faces`` (for voxel and 6 facewise neighbors, only),\n``edges`` (for voxel and 18 face- and edge-wise neighbors),\n``vertices`` (for voxel and 26 face-, edge-, and node-wise neighbors).')
    sphere = traits.Float(argstr='-neigh_RAD %s', xor=['neighborhood', 'ellipsoid'], desc="\\\nFor additional voxelwise neighborhood control, the\nradius R of a desired neighborhood can be put in; R is\na floating point number, and must be >1. Examples of\nthe numbers of voxels in a given radius are as follows\n(you can roughly approximate with the ol' :math:`4\\pi\\,R^3/3`\nthing):\n\n    * R=2.0 -> V=33\n    * R=2.3 -> V=57,\n    * R=2.9 -> V=93,\n    * R=3.1 -> V=123,\n    * R=3.9 -> V=251,\n    * R=4.5 -> V=389,\n    * R=6.1 -> V=949,\n\nbut you can choose most any value.")
    ellipsoid = traits.Tuple(traits.Float, traits.Float, traits.Float, xor=['sphere', 'neighborhood'], argstr='-neigh_X %s -neigh_Y %s -neigh_Z %s', desc="\\\nTuple indicating the x, y, and z radius of an ellipsoid\ndefining the neighbourhood of each voxel.\nThe 'hood is then made according to the following relation:\n:math:`(i/A)^2 + (j/B)^2 + (k/C)^2 \\le 1.`\nwhich will have approx. :math:`V=4 \\pi \\, A B C/3`. The impetus for\nthis freedom was for use with data having anisotropic\nvoxel edge lengths.")
    label_set = File(exists=True, argstr='-in_rois %s', desc='a set of ROIs, each labelled with distinct integers. ReHo will then be calculated per ROI.')
    overwrite = traits.Bool(desc='overwrite output file if it already exists', argstr='-overwrite')