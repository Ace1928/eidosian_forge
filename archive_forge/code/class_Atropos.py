import os
from glob import glob
from ...external.due import BibTeX
from ...utils.filemanip import split_filename, copyfile, which, fname_presuffix
from ..base import TraitedSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
from ..mixins import CopyHeaderInterface
from .base import ANTSCommand, ANTSCommandInputSpec
class Atropos(ANTSCommand):
    """
    A multivariate n-class segmentation algorithm.

    A finite mixture modeling (FMM) segmentation approach with possibilities for
    specifying prior constraints. These prior constraints include the specification
    of a prior label image, prior probability images (one for each class), and/or an
    MRF prior to enforce spatial smoothing of the labels. Similar algorithms include
    FAST and SPM.

    Examples
    --------
    >>> from nipype.interfaces.ants import Atropos
    >>> at = Atropos(
    ...     dimension=3, intensity_images='structural.nii', mask_image='mask.nii',
    ...     number_of_tissue_classes=2, likelihood_model='Gaussian', save_posteriors=True,
    ...     mrf_smoothing_factor=0.2, mrf_radius=[1, 1, 1], icm_use_synchronous_update=True,
    ...     maximum_number_of_icm_terations=1, n_iterations=5, convergence_threshold=0.000001,
    ...     posterior_formulation='Socrates', use_mixture_model_proportions=True)
    >>> at.inputs.initialization = 'Random'
    >>> at.cmdline
    'Atropos --image-dimensionality 3 --icm [1,1]
    --initialization Random[2] --intensity-image structural.nii
    --likelihood-model Gaussian --mask-image mask.nii --mrf [0.2,1x1x1] --convergence [5,1e-06]
    --output [structural_labeled.nii,POSTERIOR_%02d.nii.gz] --posterior-formulation Socrates[1]
    --use-random-seed 1'

    >>> at = Atropos(
    ...     dimension=3, intensity_images='structural.nii', mask_image='mask.nii',
    ...     number_of_tissue_classes=2, likelihood_model='Gaussian', save_posteriors=True,
    ...     mrf_smoothing_factor=0.2, mrf_radius=[1, 1, 1], icm_use_synchronous_update=True,
    ...     maximum_number_of_icm_terations=1, n_iterations=5, convergence_threshold=0.000001,
    ...     posterior_formulation='Socrates', use_mixture_model_proportions=True)
    >>> at.inputs.initialization = 'KMeans'
    >>> at.inputs.kmeans_init_centers = [100, 200]
    >>> at.cmdline
    'Atropos --image-dimensionality 3 --icm [1,1]
    --initialization KMeans[2,100,200] --intensity-image structural.nii
    --likelihood-model Gaussian --mask-image mask.nii --mrf [0.2,1x1x1] --convergence [5,1e-06]
    --output [structural_labeled.nii,POSTERIOR_%02d.nii.gz] --posterior-formulation Socrates[1]
    --use-random-seed 1'

    >>> at = Atropos(
    ...     dimension=3, intensity_images='structural.nii', mask_image='mask.nii',
    ...     number_of_tissue_classes=2, likelihood_model='Gaussian', save_posteriors=True,
    ...     mrf_smoothing_factor=0.2, mrf_radius=[1, 1, 1], icm_use_synchronous_update=True,
    ...     maximum_number_of_icm_terations=1, n_iterations=5, convergence_threshold=0.000001,
    ...     posterior_formulation='Socrates', use_mixture_model_proportions=True)
    >>> at.inputs.initialization = 'PriorProbabilityImages'
    >>> at.inputs.prior_image = 'BrainSegmentationPrior%02d.nii.gz'
    >>> at.inputs.prior_weighting = 0.8
    >>> at.inputs.prior_probability_threshold = 0.0000001
    >>> at.cmdline
    'Atropos --image-dimensionality 3 --icm [1,1]
    --initialization PriorProbabilityImages[2,BrainSegmentationPrior%02d.nii.gz,0.8,1e-07]
    --intensity-image structural.nii --likelihood-model Gaussian --mask-image mask.nii
    --mrf [0.2,1x1x1] --convergence [5,1e-06]
    --output [structural_labeled.nii,POSTERIOR_%02d.nii.gz]
    --posterior-formulation Socrates[1] --use-random-seed 1'

    >>> at = Atropos(
    ...     dimension=3, intensity_images='structural.nii', mask_image='mask.nii',
    ...     number_of_tissue_classes=2, likelihood_model='Gaussian', save_posteriors=True,
    ...     mrf_smoothing_factor=0.2, mrf_radius=[1, 1, 1], icm_use_synchronous_update=True,
    ...     maximum_number_of_icm_terations=1, n_iterations=5, convergence_threshold=0.000001,
    ...     posterior_formulation='Socrates', use_mixture_model_proportions=True)
    >>> at.inputs.initialization = 'PriorLabelImage'
    >>> at.inputs.prior_image = 'segmentation0.nii.gz'
    >>> at.inputs.number_of_tissue_classes = 2
    >>> at.inputs.prior_weighting = 0.8
    >>> at.cmdline
    'Atropos --image-dimensionality 3 --icm [1,1]
    --initialization PriorLabelImage[2,segmentation0.nii.gz,0.8] --intensity-image structural.nii
    --likelihood-model Gaussian --mask-image mask.nii --mrf [0.2,1x1x1] --convergence [5,1e-06]
    --output [structural_labeled.nii,POSTERIOR_%02d.nii.gz] --posterior-formulation Socrates[1]
    --use-random-seed 1'

    """
    input_spec = AtroposInputSpec
    output_spec = AtroposOutputSpec
    _cmd = 'Atropos'

    def _format_arg(self, opt, spec, val):
        if opt == 'initialization':
            n_classes = self.inputs.number_of_tissue_classes
            brackets = ['%d' % n_classes]
            if val == 'KMeans' and isdefined(self.inputs.kmeans_init_centers):
                centers = sorted(set(self.inputs.kmeans_init_centers))
                if len(centers) != n_classes:
                    raise ValueError('KMeans initialization with initial cluster centers requires the number of centers to match number_of_tissue_classes')
                brackets += ['%g' % c for c in centers]
            if val in ('PriorProbabilityImages', 'PriorLabelImage'):
                if not isdefined(self.inputs.prior_image) or not isdefined(self.inputs.prior_weighting):
                    raise ValueError("'%s' initialization requires setting prior_image and prior_weighting" % val)
                priors_paths = [self.inputs.prior_image]
                if '%02d' in priors_paths[0]:
                    if val == 'PriorLabelImage':
                        raise ValueError("'PriorLabelImage' initialization does not accept patterns for prior_image.")
                    priors_paths = [priors_paths[0] % i for i in range(1, n_classes + 1)]
                if not all([os.path.exists(p) for p in priors_paths]):
                    raise FileNotFoundError('One or more prior images do not exist: %s.' % ', '.join(priors_paths))
                brackets += [self.inputs.prior_image, '%g' % self.inputs.prior_weighting]
                if val == 'PriorProbabilityImages' and isdefined(self.inputs.prior_probability_threshold):
                    brackets.append('%g' % self.inputs.prior_probability_threshold)
            return '--initialization %s[%s]' % (val, ','.join(brackets))
        if opt == 'mrf_smoothing_factor':
            retval = '--mrf [%g' % val
            if isdefined(self.inputs.mrf_radius):
                retval += ',%s' % self._format_xarray([str(s) for s in self.inputs.mrf_radius])
            return retval + ']'
        if opt == 'icm_use_synchronous_update':
            retval = '--icm [%d' % val
            if isdefined(self.inputs.maximum_number_of_icm_terations):
                retval += ',%g' % self.inputs.maximum_number_of_icm_terations
            return retval + ']'
        if opt == 'n_iterations':
            retval = '--convergence [%d' % val
            if isdefined(self.inputs.convergence_threshold):
                retval += ',%g' % self.inputs.convergence_threshold
            return retval + ']'
        if opt == 'posterior_formulation':
            retval = '--posterior-formulation %s' % val
            if isdefined(self.inputs.use_mixture_model_proportions):
                retval += '[%d]' % self.inputs.use_mixture_model_proportions
            return retval
        if opt == 'out_classified_image_name':
            retval = '--output [%s' % val
            if isdefined(self.inputs.save_posteriors):
                retval += ',%s' % self.inputs.output_posteriors_name_template
            return retval + ']'
        return super(Atropos, self)._format_arg(opt, spec, val)

    def _gen_filename(self, name):
        if name == 'out_classified_image_name':
            output = self.inputs.out_classified_image_name
            if not isdefined(output):
                _, name, ext = split_filename(self.inputs.intensity_images[0])
                output = name + '_labeled' + ext
            return output

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['classified_image'] = os.path.abspath(self._gen_filename('out_classified_image_name'))
        if isdefined(self.inputs.save_posteriors) and self.inputs.save_posteriors:
            outputs['posteriors'] = []
            for i in range(self.inputs.number_of_tissue_classes):
                outputs['posteriors'].append(os.path.abspath(self.inputs.output_posteriors_name_template % (i + 1)))
        return outputs