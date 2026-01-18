import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class UnifizeInputSpec(AFNICommandInputSpec):
    in_file = File(desc='input file to 3dUnifize', argstr='-input %s', position=-1, mandatory=True, exists=True, copyfile=False)
    out_file = File(name_template='%s_unifized', desc='output image file name', argstr='-prefix %s', name_source='in_file')
    t2 = traits.Bool(desc='Treat the input as if it were T2-weighted, rather than T1-weighted. This processing is done simply by inverting the image contrast, processing it as if that result were T1-weighted, and then re-inverting the results counts of voxel overlap, i.e., each voxel will contain the number of masks that it is set in.', argstr='-T2')
    gm = traits.Bool(desc="Also scale to unifize 'gray matter' = lower intensity voxels (to aid in registering images from different scanners).", argstr='-GM')
    urad = traits.Float(desc='Sets the radius (in voxels) of the ball used for the sneaky trick. Default value is 18.3, and should be changed proportionally if the dataset voxel size differs significantly from 1 mm.', argstr='-Urad %s')
    scale_file = File(desc='output file name to save the scale factor used at each voxel ', argstr='-ssave %s')
    no_duplo = traits.Bool(desc="Do NOT use the 'duplo down' step; this can be useful for lower resolution datasets.", argstr='-noduplo')
    epi = traits.Bool(desc="Assume the input dataset is a T2 (or T2\\*) weighted EPI time series. After computing the scaling, apply it to ALL volumes (TRs) in the input dataset. That is, a given voxel will be scaled by the same factor at each TR. This option also implies '-noduplo' and '-T2'.This option turns off '-GM' if you turned it on.", argstr='-EPI', requires=['no_duplo', 't2'], xor=['gm'])
    rbt = traits.Tuple(traits.Float(), traits.Float(), traits.Float(), desc="Option for AFNI experts only.Specify the 3 parameters for the algorithm:\nR = radius; same as given by option '-Urad', [default=18.3]\nb = bottom percentile of normalizing data range, [default=70.0]\nr = top percentile of normalizing data range, [default=80.0]\n", argstr='-rbt %f %f %f')
    t2_up = traits.Float(desc='Option for AFNI experts only.Set the upper percentile point used for T2-T1 inversion. Allowed to be anything between 90 and 100 (inclusive), with default to 98.5  (for no good reason).', argstr='-T2up %f')
    cl_frac = traits.Float(desc="Option for AFNI experts only.Set the automask 'clip level fraction'. Must be between 0.1 and 0.9. A small fraction means to make the initial threshold for clipping (a la 3dClipLevel) smaller, which will tend to make the mask larger.  [default=0.1]", argstr='-clfrac %f')
    quiet = traits.Bool(desc="Don't print the progress messages.", argstr='-quiet')