import os
from glob import glob
from ...external.due import BibTeX
from ...utils.filemanip import split_filename, copyfile, which, fname_presuffix
from ..base import TraitedSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
from ..mixins import CopyHeaderInterface
from .base import ANTSCommand, ANTSCommandInputSpec
class JointFusionInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, 4, argstr='-d %d', desc='This option forces the image to be treated as a specified-dimensional image. If not specified, the program tries to infer the dimensionality from the input image.')
    target_image = traits.List(InputMultiPath(File(exists=True)), argstr='-t %s', mandatory=True, desc='The target image (or multimodal target images) assumed to be aligned to a common image domain.')
    atlas_image = traits.List(InputMultiPath(File(exists=True)), argstr='-g %s...', mandatory=True, desc='The atlas image (or multimodal atlas images) assumed to be aligned to a common image domain.')
    atlas_segmentation_image = InputMultiPath(File(exists=True), argstr='-l %s...', mandatory=True, desc='The atlas segmentation images. For performing label fusion the number of specified segmentations should be identical to the number of atlas image sets.')
    alpha = traits.Float(default_value=0.1, usedefault=True, argstr='-a %s', desc='Regularization term added to matrix Mx for calculating the inverse. Default = 0.1')
    beta = traits.Float(default_value=2.0, usedefault=True, argstr='-b %s', desc='Exponent for mapping intensity difference to the joint error. Default = 2.0')
    retain_label_posterior_images = traits.Bool(False, argstr='-r', usedefault=True, requires=['atlas_segmentation_image'], desc='Retain label posterior probability images. Requires atlas segmentations to be specified. Default = false')
    retain_atlas_voting_images = traits.Bool(False, argstr='-f', usedefault=True, desc='Retain atlas voting images. Default = false')
    constrain_nonnegative = traits.Bool(False, argstr='-c', usedefault=True, desc='Constrain solution to non-negative weights.')
    patch_radius = traits.ListInt(minlen=3, maxlen=3, argstr='-p %s', desc='Patch radius for similarity measures. Default: 2x2x2')
    patch_metric = traits.Enum('PC', 'MSQ', argstr='-m %s', desc="Metric to be used in determining the most similar neighborhood patch. Options include Pearson's correlation (PC) and mean squares (MSQ). Default = PC (Pearson correlation).")
    search_radius = traits.List([3, 3, 3], minlen=1, maxlen=3, argstr='-s %s', usedefault=True, desc='Search radius for similarity measures. Default = 3x3x3. One can also specify an image where the value at the voxel specifies the isotropic search radius at that voxel.')
    exclusion_image_label = traits.List(traits.Str(), argstr='-e %s', requires=['exclusion_image'], desc='Specify a label for the exclusion region.')
    exclusion_image = traits.List(File(exists=True), desc='Specify an exclusion region for the given label.')
    mask_image = File(argstr='-x %s', exists=True, desc='If a mask image is specified, fusion is only performed in the mask region.')
    out_label_fusion = File(argstr='%s', hash_files=False, desc='The output label fusion image.')
    out_intensity_fusion_name_format = traits.Str(argstr='', desc='Optional intensity fusion image file name format. (e.g. "antsJointFusionIntensity_%d.nii.gz")')
    out_label_post_prob_name_format = traits.Str('antsJointFusionPosterior_%d.nii.gz', requires=['out_label_fusion', 'out_intensity_fusion_name_format'], desc='Optional label posterior probability image file name format.')
    out_atlas_voting_weight_name_format = traits.Str('antsJointFusionVotingWeight_%d.nii.gz', requires=['out_label_fusion', 'out_intensity_fusion_name_format', 'out_label_post_prob_name_format'], desc='Optional atlas voting weight image file name format.')
    verbose = traits.Bool(False, argstr='-v', desc='Verbose output.')