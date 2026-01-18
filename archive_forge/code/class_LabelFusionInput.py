import os
import warnings
from ..base import (
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
from ...utils.filemanip import load_json, save_json, split_filename
class LabelFusionInput(CommandLineInputSpec):
    """Input Spec for LabelFusion."""
    in_file = File(argstr='-in %s', exists=True, mandatory=True, position=1, desc='Filename of the 4D integer label image.')
    template_file = File(exists=True, desc='Registered templates (4D Image)')
    file_to_seg = File(exists=True, mandatory=True, desc='Original image to segment (3D Image)')
    mask_file = File(argstr='-mask %s', exists=True, desc='Filename of the ROI for label fusion')
    out_file = File(argstr='-out %s', name_source=['in_file'], name_template='%s', desc='Output consensus segmentation')
    prob_flag = traits.Bool(desc='Probabilistic/Fuzzy segmented image', argstr='-outProb')
    desc = 'Verbose level [0 = off, 1 = on, 2 = debug] (default = 0)'
    verbose = traits.Enum('0', '1', '2', desc=desc, argstr='-v %s')
    desc = 'Only consider non-consensus voxels to calculate statistics'
    unc = traits.Bool(desc=desc, argstr='-unc')
    classifier_type = traits.Enum('STEPS', 'STAPLE', 'MV', 'SBA', argstr='-%s', mandatory=True, position=2, desc='Type of Classifier Fusion.')
    desc = 'Gaussian kernel size in mm to compute the local similarity'
    kernel_size = traits.Float(desc=desc)
    template_num = traits.Int(desc='Number of labels to use')
    sm_ranking = traits.Enum('ALL', 'GNCC', 'ROINCC', 'LNCC', argstr='-%s', usedefault=True, position=3, desc='Ranking for STAPLE and MV')
    dilation_roi = traits.Int(desc='Dilation of the ROI ( <int> d>=1 )')
    desc = 'Proportion of the label (only for single labels).'
    proportion = traits.Float(argstr='-prop %s', desc=desc)
    desc = 'Update label proportions at each iteration'
    prob_update_flag = traits.Bool(desc=desc, argstr='-prop_update')
    desc = 'Value of P and Q [ 0 < (P,Q) < 1 ] (default = 0.99 0.99)'
    set_pq = traits.Tuple(traits.Float, traits.Float, argstr='-setPQ %f %f', desc=desc)
    mrf_value = traits.Float(argstr='-MRF_beta %f', desc='MRF prior strength (between 0 and 5)')
    desc = 'Maximum number of iterations (default = 15).'
    max_iter = traits.Int(argstr='-max_iter %d', desc=desc)
    desc = 'If <float> percent of labels agree, then area is not uncertain.'
    unc_thresh = traits.Float(argstr='-uncthres %f', desc=desc)
    desc = 'Ratio for convergence (default epsilon = 10^-5).'
    conv = traits.Float(argstr='-conv %f', desc=desc)