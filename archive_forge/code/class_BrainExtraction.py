import os
from glob import glob
from ...external.due import BibTeX
from ...utils.filemanip import split_filename, copyfile, which, fname_presuffix
from ..base import TraitedSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
from ..mixins import CopyHeaderInterface
from .base import ANTSCommand, ANTSCommandInputSpec
class BrainExtraction(ANTSCommand):
    """
    Atlas-based brain extraction.

    Examples
    --------
    >>> from nipype.interfaces.ants.segmentation import BrainExtraction
    >>> brainextraction = BrainExtraction()
    >>> brainextraction.inputs.dimension = 3
    >>> brainextraction.inputs.anatomical_image ='T1.nii.gz'
    >>> brainextraction.inputs.brain_template = 'study_template.nii.gz'
    >>> brainextraction.inputs.brain_probability_mask ='ProbabilityMaskOfStudyTemplate.nii.gz'
    >>> brainextraction.cmdline
    'antsBrainExtraction.sh -a T1.nii.gz -m ProbabilityMaskOfStudyTemplate.nii.gz
    -e study_template.nii.gz -d 3 -s nii.gz -o highres001_'

    """
    input_spec = BrainExtractionInputSpec
    output_spec = BrainExtractionOutputSpec
    _cmd = 'antsBrainExtraction.sh'

    def _run_interface(self, runtime, correct_return_codes=(0,)):
        out_environ = self._get_environ()
        ants_path = out_environ.get('ANTSPATH', None) or os.getenv('ANTSPATH', None)
        if ants_path is None:
            cmd_path = which('antsRegistration', env=runtime.environ)
            if not cmd_path:
                raise RuntimeError('The environment variable $ANTSPATH is not defined in host "%s", and Nipype could not determine it automatically.' % runtime.hostname)
            ants_path = os.path.dirname(cmd_path)
        self.inputs.environ.update({'ANTSPATH': ants_path})
        runtime.environ.update({'ANTSPATH': ants_path})
        runtime = super(BrainExtraction, self)._run_interface(runtime)
        if 'we cant find' in runtime.stdout:
            for line in runtime.stdout.split('\n'):
                if line.strip().startswith('we cant find'):
                    tool = line.strip().replace('we cant find the', '').split(' ')[0]
                    break
            errmsg = 'antsBrainExtraction.sh requires "%s" to be found in $ANTSPATH ($ANTSPATH="%s").' % (tool, ants_path)
            if runtime.stderr is None:
                runtime.stderr = errmsg
            else:
                runtime.stderr += '\n' + errmsg
            runtime.returncode = 1
            self.raise_exception(runtime)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['BrainExtractionMask'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'BrainExtractionMask.' + self.inputs.image_suffix)
        outputs['BrainExtractionBrain'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'BrainExtractionBrain.' + self.inputs.image_suffix)
        if isdefined(self.inputs.keep_temporary_files) and self.inputs.keep_temporary_files != 0:
            outputs['BrainExtractionCSF'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'BrainExtractionCSF.' + self.inputs.image_suffix)
            outputs['BrainExtractionGM'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'BrainExtractionGM.' + self.inputs.image_suffix)
            outputs['BrainExtractionInitialAffine'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'BrainExtractionInitialAffine.mat')
            outputs['BrainExtractionInitialAffineFixed'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'BrainExtractionInitialAffineFixed.' + self.inputs.image_suffix)
            outputs['BrainExtractionInitialAffineMoving'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'BrainExtractionInitialAffineMoving.' + self.inputs.image_suffix)
            outputs['BrainExtractionLaplacian'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'BrainExtractionLaplacian.' + self.inputs.image_suffix)
            outputs['BrainExtractionPrior0GenericAffine'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'BrainExtractionPrior0GenericAffine.mat')
            outputs['BrainExtractionPrior1InverseWarp'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'BrainExtractionPrior1InverseWarp.' + self.inputs.image_suffix)
            outputs['BrainExtractionPrior1Warp'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'BrainExtractionPrior1Warp.' + self.inputs.image_suffix)
            outputs['BrainExtractionPriorWarped'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'BrainExtractionPriorWarped.' + self.inputs.image_suffix)
            outputs['BrainExtractionSegmentation'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'BrainExtractionSegmentation.' + self.inputs.image_suffix)
            outputs['BrainExtractionTemplateLaplacian'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'BrainExtractionTemplateLaplacian.' + self.inputs.image_suffix)
            outputs['BrainExtractionTmp'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'BrainExtractionTmp.' + self.inputs.image_suffix)
            outputs['BrainExtractionWM'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'BrainExtractionWM.' + self.inputs.image_suffix)
            outputs['N4Corrected0'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'N4Corrected0.' + self.inputs.image_suffix)
            outputs['N4Truncated0'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'N4Truncated0.' + self.inputs.image_suffix)
        return outputs