from builtins import range
import os
from glob import glob
from .base import ANTSCommand, ANTSCommandInputSpec
from ..base import TraitedSpec, File, traits, isdefined, OutputMultiPath
from ...utils.filemanip import split_filename
class antsIntroductionInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, argstr='-d %d', usedefault=True, desc='image dimension (2 or 3)', position=1)
    reference_image = File(exists=True, argstr='-r %s', desc='template file to warp to', mandatory=True, copyfile=True)
    input_image = File(exists=True, argstr='-i %s', desc='input image to warp to template', mandatory=True, copyfile=False)
    force_proceed = traits.Bool(argstr='-f 1', desc='force script to proceed even if headers may be incompatible')
    inverse_warp_template_labels = traits.Bool(argstr='-l', desc='Applies inverse warp to the template labels to estimate label positions in target space (use for template-based segmentation)')
    max_iterations = traits.List(traits.Int, argstr='-m %s', sep='x', desc='maximum number of iterations (must be list of integers in the form [J,K,L...]: J = coarsest resolution iterations, K = middle resolution iterations, L = fine resolution iterations')
    bias_field_correction = traits.Bool(argstr='-n 1', desc='Applies bias field correction to moving image')
    similarity_metric = traits.Enum('PR', 'CC', 'MI', 'MSQ', argstr='-s %s', desc='Type of similartiy metric used for registration (CC = cross correlation, MI = mutual information, PR = probability mapping, MSQ = mean square difference)')
    transformation_model = traits.Enum('GR', 'EL', 'SY', 'S2', 'EX', 'DD', 'RI', 'RA', argstr='-t %s', usedefault=True, desc='Type of transofmration model used for registration (EL = elastic transformation model, SY = SyN with time, arbitrary number of time points, S2 =  SyN with time optimized for 2 time points, GR = greedy SyN, EX = exponential, DD = diffeomorphic demons style exponential mapping, RI = purely rigid, RA = affine rigid')
    out_prefix = traits.Str('ants_', argstr='-o %s', usedefault=True, desc='Prefix that is prepended to all output files (default = ants_)')
    quality_check = traits.Bool(argstr='-q 1', desc='Perform a quality check of the result')