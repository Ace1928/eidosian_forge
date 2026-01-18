import os
from glob import glob
from ...external.due import BibTeX
from ...utils.filemanip import split_filename, copyfile, which, fname_presuffix
from ..base import TraitedSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
from ..mixins import CopyHeaderInterface
from .base import ANTSCommand, ANTSCommandInputSpec
class JointFusion(ANTSCommand):
    """
    An image fusion algorithm.

    Developed by Hongzhi Wang and Paul Yushkevich, and it won segmentation challenges
    at MICCAI 2012 and MICCAI 2013.
    The original label fusion framework was extended to accommodate intensities by Brian
    Avants.
    This implementation is based on Paul's original ITK-style implementation
    and Brian's ANTsR implementation.

    References include 1) H. Wang, J. W. Suh, S.
    Das, J. Pluta, C. Craige, P. Yushkevich, Multi-atlas segmentation with joint
    label fusion IEEE Trans. on Pattern Analysis and Machine Intelligence, 35(3),
    611-623, 2013. and 2) H. Wang and P. A. Yushkevich, Multi-atlas segmentation
    with joint label fusion and corrective learning--an open source implementation,
    Front. Neuroinform., 2013.

    Examples
    --------
    >>> from nipype.interfaces.ants import JointFusion
    >>> jf = JointFusion()
    >>> jf.inputs.out_label_fusion = 'ants_fusion_label_output.nii'
    >>> jf.inputs.atlas_image = [ ['rc1s1.nii','rc1s2.nii'] ]
    >>> jf.inputs.atlas_segmentation_image = ['segmentation0.nii.gz']
    >>> jf.inputs.target_image = ['im1.nii']
    >>> jf.cmdline
    "antsJointFusion -a 0.1 -g ['rc1s1.nii', 'rc1s2.nii'] -l segmentation0.nii.gz
    -b 2.0 -o ants_fusion_label_output.nii -s 3x3x3 -t ['im1.nii']"

    >>> jf.inputs.target_image = [ ['im1.nii', 'im2.nii'] ]
    >>> jf.cmdline
    "antsJointFusion -a 0.1 -g ['rc1s1.nii', 'rc1s2.nii'] -l segmentation0.nii.gz
    -b 2.0 -o ants_fusion_label_output.nii -s 3x3x3 -t ['im1.nii', 'im2.nii']"

    >>> jf.inputs.atlas_image = [ ['rc1s1.nii','rc1s2.nii'],
    ...                                        ['rc2s1.nii','rc2s2.nii'] ]
    >>> jf.inputs.atlas_segmentation_image = ['segmentation0.nii.gz',
    ...                                                    'segmentation1.nii.gz']
    >>> jf.cmdline
    "antsJointFusion -a 0.1 -g ['rc1s1.nii', 'rc1s2.nii'] -g ['rc2s1.nii', 'rc2s2.nii']
    -l segmentation0.nii.gz -l segmentation1.nii.gz -b 2.0 -o ants_fusion_label_output.nii
    -s 3x3x3 -t ['im1.nii', 'im2.nii']"

    >>> jf.inputs.dimension = 3
    >>> jf.inputs.alpha = 0.5
    >>> jf.inputs.beta = 1.0
    >>> jf.inputs.patch_radius = [3,2,1]
    >>> jf.inputs.search_radius = [3]
    >>> jf.cmdline
    "antsJointFusion -a 0.5 -g ['rc1s1.nii', 'rc1s2.nii'] -g ['rc2s1.nii', 'rc2s2.nii']
    -l segmentation0.nii.gz -l segmentation1.nii.gz -b 1.0 -d 3 -o ants_fusion_label_output.nii
    -p 3x2x1 -s 3 -t ['im1.nii', 'im2.nii']"

    >>> jf.inputs.search_radius = ['mask.nii']
    >>> jf.inputs.verbose = True
    >>> jf.inputs.exclusion_image = ['roi01.nii', 'roi02.nii']
    >>> jf.inputs.exclusion_image_label = ['1','2']
    >>> jf.cmdline
    "antsJointFusion -a 0.5 -g ['rc1s1.nii', 'rc1s2.nii'] -g ['rc2s1.nii', 'rc2s2.nii']
    -l segmentation0.nii.gz -l segmentation1.nii.gz -b 1.0 -d 3 -e 1[roi01.nii] -e 2[roi02.nii]
    -o ants_fusion_label_output.nii -p 3x2x1 -s mask.nii -t ['im1.nii', 'im2.nii'] -v"

    >>> jf.inputs.out_label_fusion = 'ants_fusion_label_output.nii'
    >>> jf.inputs.out_intensity_fusion_name_format = 'ants_joint_fusion_intensity_%d.nii.gz'
    >>> jf.inputs.out_label_post_prob_name_format = 'ants_joint_fusion_posterior_%d.nii.gz'
    >>> jf.inputs.out_atlas_voting_weight_name_format = 'ants_joint_fusion_voting_weight_%d.nii.gz'
    >>> jf.cmdline
    "antsJointFusion -a 0.5 -g ['rc1s1.nii', 'rc1s2.nii'] -g ['rc2s1.nii', 'rc2s2.nii']
    -l segmentation0.nii.gz -l segmentation1.nii.gz -b 1.0 -d 3 -e 1[roi01.nii] -e 2[roi02.nii]
    -o [ants_fusion_label_output.nii, ants_joint_fusion_intensity_%d.nii.gz,
    ants_joint_fusion_posterior_%d.nii.gz, ants_joint_fusion_voting_weight_%d.nii.gz]
    -p 3x2x1 -s mask.nii -t ['im1.nii', 'im2.nii'] -v"

    """
    input_spec = JointFusionInputSpec
    output_spec = JointFusionOutputSpec
    _cmd = 'antsJointFusion'

    def _format_arg(self, opt, spec, val):
        if opt == 'exclusion_image_label':
            retval = []
            for ii in range(len(self.inputs.exclusion_image_label)):
                retval.append('-e {0}[{1}]'.format(self.inputs.exclusion_image_label[ii], self.inputs.exclusion_image[ii]))
            return ' '.join(retval)
        if opt == 'patch_radius':
            return '-p {0}'.format(self._format_xarray(val))
        if opt == 'search_radius':
            return '-s {0}'.format(self._format_xarray(val))
        if opt == 'out_label_fusion':
            args = [self.inputs.out_label_fusion]
            for option in (self.inputs.out_intensity_fusion_name_format, self.inputs.out_label_post_prob_name_format, self.inputs.out_atlas_voting_weight_name_format):
                if isdefined(option):
                    args.append(option)
                else:
                    break
            if len(args) == 1:
                return ' '.join(('-o', args[0]))
            return '-o [{}]'.format(', '.join(args))
        if opt == 'out_intensity_fusion_name_format':
            if not isdefined(self.inputs.out_label_fusion):
                return '-o {0}'.format(self.inputs.out_intensity_fusion_name_format)
            return ''
        if opt == 'atlas_image':
            return ' '.join(['-g [{0}]'.format(', '.join(("'%s'" % fn for fn in ai))) for ai in self.inputs.atlas_image])
        if opt == 'target_image':
            return ' '.join(['-t [{0}]'.format(', '.join(("'%s'" % fn for fn in ai))) for ai in self.inputs.target_image])
        if opt == 'atlas_segmentation_image':
            if len(val) != len(self.inputs.atlas_image):
                raise ValueError('Number of specified segmentations should be identical to the number of atlas image sets {0}!={1}'.format(len(val), len(self.inputs.atlas_image)))
            return ' '.join(['-l {0}'.format(fn) for fn in self.inputs.atlas_segmentation_image])
        return super(AntsJointFusion, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = self._outputs().get()
        if isdefined(self.inputs.out_label_fusion):
            outputs['out_label_fusion'] = os.path.abspath(self.inputs.out_label_fusion)
        if isdefined(self.inputs.out_intensity_fusion_name_format):
            outputs['out_intensity_fusion'] = glob(os.path.abspath(self.inputs.out_intensity_fusion_name_format.replace('%d', '*')))
        if isdefined(self.inputs.out_label_post_prob_name_format):
            outputs['out_label_post_prob'] = glob(os.path.abspath(self.inputs.out_label_post_prob_name_format.replace('%d', '*')))
        if isdefined(self.inputs.out_atlas_voting_weight_name_format):
            outputs['out_atlas_voting_weight'] = glob(os.path.abspath(self.inputs.out_atlas_voting_weight_name_format.replace('%d', '*')))
        return outputs