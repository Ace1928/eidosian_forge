import os
from glob import glob
from ...external.due import BibTeX
from ...utils.filemanip import split_filename, copyfile, which, fname_presuffix
from ..base import TraitedSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
from ..mixins import CopyHeaderInterface
from .base import ANTSCommand, ANTSCommandInputSpec
class AtroposInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, 4, argstr='--image-dimensionality %d', usedefault=True, desc='image dimension (2, 3, or 4)')
    intensity_images = InputMultiPath(File(exists=True), argstr='--intensity-image %s...', mandatory=True)
    mask_image = File(exists=True, argstr='--mask-image %s', mandatory=True)
    initialization = traits.Enum('Random', 'Otsu', 'KMeans', 'PriorProbabilityImages', 'PriorLabelImage', argstr='%s', requires=['number_of_tissue_classes'], mandatory=True)
    kmeans_init_centers = traits.List(traits.Either(traits.Int, traits.Float), minlen=1)
    prior_image = traits.Either(File(exists=True), traits.Str, desc="either a string pattern (e.g., 'prior%02d.nii') or an existing vector-image file.")
    number_of_tissue_classes = traits.Int(mandatory=True)
    prior_weighting = traits.Float()
    prior_probability_threshold = traits.Float(requires=['prior_weighting'])
    likelihood_model = traits.Str(argstr='--likelihood-model %s')
    mrf_smoothing_factor = traits.Float(argstr='%s')
    mrf_radius = traits.List(traits.Int(), requires=['mrf_smoothing_factor'])
    icm_use_synchronous_update = traits.Bool(argstr='%s')
    maximum_number_of_icm_terations = traits.Int(requires=['icm_use_synchronous_update'])
    n_iterations = traits.Int(argstr='%s')
    convergence_threshold = traits.Float(requires=['n_iterations'])
    posterior_formulation = traits.Str(argstr='%s')
    use_random_seed = traits.Bool(True, argstr='--use-random-seed %d', desc='use random seed value over constant', usedefault=True)
    use_mixture_model_proportions = traits.Bool(requires=['posterior_formulation'])
    out_classified_image_name = File(argstr='%s', genfile=True, hash_files=False)
    save_posteriors = traits.Bool()
    output_posteriors_name_template = traits.Str('POSTERIOR_%02d.nii.gz', usedefault=True)