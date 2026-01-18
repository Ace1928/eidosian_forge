import os
from glob import glob
from ...external.due import BibTeX
from ...utils.filemanip import split_filename, copyfile, which, fname_presuffix
from ..base import TraitedSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
from ..mixins import CopyHeaderInterface
from .base import ANTSCommand, ANTSCommandInputSpec
class CorticalThickness(ANTSCommand):
    """
    Examples
    --------
    >>> from nipype.interfaces.ants.segmentation import CorticalThickness
    >>> corticalthickness = CorticalThickness()
    >>> corticalthickness.inputs.dimension = 3
    >>> corticalthickness.inputs.anatomical_image ='T1.nii.gz'
    >>> corticalthickness.inputs.brain_template = 'study_template.nii.gz'
    >>> corticalthickness.inputs.brain_probability_mask ='ProbabilityMaskOfStudyTemplate.nii.gz'
    >>> corticalthickness.inputs.segmentation_priors = ['BrainSegmentationPrior01.nii.gz',
    ...                                                 'BrainSegmentationPrior02.nii.gz',
    ...                                                 'BrainSegmentationPrior03.nii.gz',
    ...                                                 'BrainSegmentationPrior04.nii.gz']
    >>> corticalthickness.inputs.t1_registration_template = 'brain_study_template.nii.gz'
    >>> corticalthickness.cmdline
    'antsCorticalThickness.sh -a T1.nii.gz -m ProbabilityMaskOfStudyTemplate.nii.gz
    -e study_template.nii.gz -d 3 -s nii.gz -o antsCT_
    -p nipype_priors/BrainSegmentationPrior%02d.nii.gz -t brain_study_template.nii.gz'

    """
    input_spec = CorticalThicknessInputSpec
    output_spec = CorticalThicknessOutputSpec
    _cmd = 'antsCorticalThickness.sh'

    def _format_arg(self, opt, spec, val):
        if opt == 'anatomical_image':
            retval = '-a %s' % val
            return retval
        if opt == 'brain_template':
            retval = '-e %s' % val
            return retval
        if opt == 'brain_probability_mask':
            retval = '-m %s' % val
            return retval
        if opt == 'out_prefix':
            retval = '-o %s' % val
            return retval
        if opt == 't1_registration_template':
            retval = '-t %s' % val
            return retval
        if opt == 'segmentation_priors':
            _, _, ext = split_filename(self.inputs.segmentation_priors[0])
            retval = '-p nipype_priors/BrainSegmentationPrior%02d' + ext
            return retval
        return super(CorticalThickness, self)._format_arg(opt, spec, val)

    def _run_interface(self, runtime, correct_return_codes=[0]):
        priors_directory = os.path.join(os.getcwd(), 'nipype_priors')
        if not os.path.exists(priors_directory):
            os.makedirs(priors_directory)
        _, _, ext = split_filename(self.inputs.segmentation_priors[0])
        for i, f in enumerate(self.inputs.segmentation_priors):
            target = os.path.join(priors_directory, 'BrainSegmentationPrior%02d' % (i + 1) + ext)
            if not (os.path.exists(target) and os.path.realpath(target) == os.path.abspath(f)):
                copyfile(os.path.abspath(f), target)
        runtime = super(CorticalThickness, self)._run_interface(runtime)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['BrainExtractionMask'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'BrainExtractionMask.' + self.inputs.image_suffix)
        outputs['ExtractedBrainN4'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'ExtractedBrain0N4.' + self.inputs.image_suffix)
        outputs['BrainSegmentation'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'BrainSegmentation.' + self.inputs.image_suffix)
        outputs['BrainSegmentationN4'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'BrainSegmentation0N4.' + self.inputs.image_suffix)
        posteriors = []
        for i in range(len(self.inputs.segmentation_priors)):
            posteriors.append(os.path.join(os.getcwd(), self.inputs.out_prefix + 'BrainSegmentationPosteriors%02d.' % (i + 1) + self.inputs.image_suffix))
        outputs['BrainSegmentationPosteriors'] = posteriors
        outputs['CorticalThickness'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'CorticalThickness.' + self.inputs.image_suffix)
        outputs['TemplateToSubject1GenericAffine'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'TemplateToSubject1GenericAffine.mat')
        outputs['TemplateToSubject0Warp'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'TemplateToSubject0Warp.' + self.inputs.image_suffix)
        outputs['SubjectToTemplate1Warp'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'SubjectToTemplate1Warp.' + self.inputs.image_suffix)
        outputs['SubjectToTemplate0GenericAffine'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'SubjectToTemplate0GenericAffine.mat')
        outputs['SubjectToTemplateLogJacobian'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'SubjectToTemplateLogJacobian.' + self.inputs.image_suffix)
        outputs['CorticalThicknessNormedToTemplate'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'CorticalThickness.' + self.inputs.image_suffix)
        outputs['BrainVolumes'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'brainvols.csv')
        return outputs