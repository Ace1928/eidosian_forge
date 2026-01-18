import os
import warnings
from ..base import (
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
from ...utils.filemanip import load_json, save_json, split_filename
class LabelFusion(NiftySegCommand):
    """Interface for executable seg_LabelFusion from NiftySeg platform using
    type STEPS as classifier Fusion.

    This executable implements 4 fusion strategies (-STEPS, -STAPLE, -MV or
    - SBA), all of them using either a global (-GNCC), ROI-based (-ROINCC),
    local (-LNCC) or no image similarity (-ALL). Combinations of fusion
    algorithms and similarity metrics give rise to different variants of known
    algorithms. As an example, using LNCC and MV as options will run a locally
    weighted voting strategy with LNCC derived weights, while using STAPLE and
    LNCC is equivalent to running STEPS as per its original formulation.
    A few other options pertaining the use of an MRF (-MRF beta), the initial
    sensitivity and specificity estimates and the use of only non-consensus
    voxels (-unc) for the STAPLE and STEPS algorithm. All processing can be
    masked (-mask), greatly reducing memory consumption.

    As an example, the command to use STEPS should be:
    seg_LabFusion -in 4D_Propragated_Labels_to_fuse.nii -out     FusedSegmentation.nii -STEPS 2 15 TargetImage.nii     4D_Propagated_Intensities.nii

    `Source code <http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftySeg>`_ |
    `Documentation <http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftySeg_documentation>`_

    Examples
    --------
    >>> from nipype.interfaces import niftyseg
    >>> node = niftyseg.LabelFusion()
    >>> node.inputs.in_file = 'im1.nii'
    >>> node.inputs.kernel_size = 2.0
    >>> node.inputs.file_to_seg = 'im2.nii'
    >>> node.inputs.template_file = 'im3.nii'
    >>> node.inputs.template_num = 2
    >>> node.inputs.classifier_type = 'STEPS'
    >>> node.cmdline
    'seg_LabFusion -in im1.nii -STEPS 2.000000 2 im2.nii im3.nii -out im1_steps.nii'

    """
    _cmd = get_custom_path('seg_LabFusion', env_dir='NIFTYSEGDIR')
    input_spec = LabelFusionInput
    output_spec = LabelFusionOutput
    _suffix = '_label_fused'

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for seg_maths."""
        if opt in ['proportion', 'prob_update_flag', 'set_pq', 'mrf_value', 'max_iter', 'unc_thresh', 'conv'] and self.inputs.classifier_type not in ['STAPLE', 'STEPS']:
            return ''
        if opt == 'sm_ranking':
            return self.get_staple_args(val)
        if opt == 'classifier_type' and val == 'STEPS':
            return self.get_steps_args()
        return super(LabelFusion, self)._format_arg(opt, spec, val)

    def get_steps_args(self):
        if not isdefined(self.inputs.template_file):
            err = "LabelFusion requires a value for input 'template_file' when 'classifier_type' is set to 'STEPS'."
            raise NipypeInterfaceError(err)
        if not isdefined(self.inputs.kernel_size):
            err = "LabelFusion requires a value for input 'kernel_size' when 'classifier_type' is set to 'STEPS'."
            raise NipypeInterfaceError(err)
        if not isdefined(self.inputs.template_num):
            err = "LabelFusion requires a value for input 'template_num' when 'classifier_type' is set to 'STEPS'."
            raise NipypeInterfaceError(err)
        return '-STEPS %f %d %s %s' % (self.inputs.kernel_size, self.inputs.template_num, self.inputs.file_to_seg, self.inputs.template_file)

    def get_staple_args(self, ranking):
        classtype = self.inputs.classifier_type
        if classtype not in ['STAPLE', 'MV']:
            return None
        if ranking == 'ALL':
            return '-ALL'
        if not isdefined(self.inputs.template_file):
            err = "LabelFusion requires a value for input 'tramplate_file' when 'classifier_type' is set to '%s' and 'sm_ranking' is set to '%s'."
            raise NipypeInterfaceError(err % (classtype, ranking))
        if not isdefined(self.inputs.template_num):
            err = "LabelFusion requires a value for input 'template-num' when 'classifier_type' is set to '%s' and 'sm_ranking' is set to '%s'."
            raise NipypeInterfaceError(err % (classtype, ranking))
        if ranking == 'GNCC':
            if not isdefined(self.inputs.template_num):
                err = "LabelFusion requires a value for input 'template_num' when 'classifier_type' is set to '%s' and 'sm_ranking' is set to '%s'."
                raise NipypeInterfaceError(err % (classtype, ranking))
            return '-%s %d %s %s' % (ranking, self.inputs.template_num, self.inputs.file_to_seg, self.inputs.template_file)
        elif ranking == 'ROINCC':
            if not isdefined(self.inputs.dilation_roi):
                err = "LabelFusion requires a value for input 'dilation_roi' when 'classifier_type' is set to '%s' and 'sm_ranking' is set to '%s'."
                raise NipypeInterfaceError(err % (classtype, ranking))
            elif self.inputs.dilation_roi < 1:
                err = "The 'dilation_roi' trait of a LabelFusionInput instance must be an integer >= 1, but a value of '%s' was specified."
                raise NipypeInterfaceError(err % self.inputs.dilation_roi)
            return '-%s %d %d %s %s' % (ranking, self.inputs.dilation_roi, self.inputs.template_num, self.inputs.file_to_seg, self.inputs.template_file)
        elif ranking == 'LNCC':
            if not isdefined(self.inputs.kernel_size):
                err = "LabelFusion requires a value for input 'kernel_size' when 'classifier_type' is set to '%s' and 'sm_ranking' is set to '%s'."
                raise NipypeInterfaceError(err % (classtype, ranking))
            return '-%s %f %d %s %s' % (ranking, self.inputs.kernel_size, self.inputs.template_num, self.inputs.file_to_seg, self.inputs.template_file)

    def _overload_extension(self, value, name=None):
        path, base, _ = split_filename(value)
        _, _, ext = split_filename(self.inputs.in_file)
        suffix = self.inputs.classifier_type.lower()
        return os.path.join(path, '{0}_{1}{2}'.format(base, suffix, ext))