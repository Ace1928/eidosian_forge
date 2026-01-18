import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class MultiChannelNewSegment(SPMCommand):
    """Use spm_preproc8 (New Segment) to separate structural images into
    different tissue classes. Supports multiple modalities and multichannel inputs.

    http://www.fil.ion.ucl.ac.uk/spm/doc/manual.pdf#page=45

    Examples
    --------
    >>> import nipype.interfaces.spm as spm
    >>> seg = spm.MultiChannelNewSegment()
    >>> seg.inputs.channels = [('structural.nii',(0.0001, 60, (True, True)))]
    >>> seg.run() # doctest: +SKIP

    For VBM pre-processing [http://www.fil.ion.ucl.ac.uk/~john/misc/VBMclass10.pdf],
    TPM.nii should be replaced by /path/to/spm8/toolbox/Seg/TPM.nii

    >>> seg = MultiChannelNewSegment()
    >>> channel1= ('T1.nii',(0.0001, 60, (True, True)))
    >>> channel2= ('T2.nii',(0.0001, 60, (True, True)))
    >>> seg.inputs.channels = [channel1, channel2]
    >>> tissue1 = (('TPM.nii', 1), 2, (True,True), (False, False))
    >>> tissue2 = (('TPM.nii', 2), 2, (True,True), (False, False))
    >>> tissue3 = (('TPM.nii', 3), 2, (True,False), (False, False))
    >>> tissue4 = (('TPM.nii', 4), 2, (False,False), (False, False))
    >>> tissue5 = (('TPM.nii', 5), 2, (False,False), (False, False))
    >>> seg.inputs.tissues = [tissue1, tissue2, tissue3, tissue4, tissue5]
    >>> seg.run() # doctest: +SKIP

    """
    input_spec = MultiChannelNewSegmentInputSpec
    output_spec = MultiChannelNewSegmentOutputSpec

    def __init__(self, **inputs):
        _local_version = SPMCommand().version
        if _local_version and '12.' in _local_version:
            self._jobtype = 'spatial'
            self._jobname = 'preproc'
        else:
            self._jobtype = 'tools'
            self._jobname = 'preproc8'
        SPMCommand.__init__(self, **inputs)

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm"""
        if opt == 'channels':
            new_channels = []
            for channel in val:
                new_channel = {}
                new_channel['vols'] = scans_for_fnames(channel[0])
                if isdefined(channel[1]):
                    info = channel[1]
                    new_channel['biasreg'] = info[0]
                    new_channel['biasfwhm'] = info[1]
                    new_channel['write'] = [int(info[2][0]), int(info[2][1])]
                new_channels.append(new_channel)
            return new_channels
        elif opt == 'tissues':
            new_tissues = []
            for tissue in val:
                new_tissue = {}
                new_tissue['tpm'] = np.array([','.join([tissue[0][0], str(tissue[0][1])])], dtype=object)
                new_tissue['ngaus'] = tissue[1]
                new_tissue['native'] = [int(tissue[2][0]), int(tissue[2][1])]
                new_tissue['warped'] = [int(tissue[3][0]), int(tissue[3][1])]
                new_tissues.append(new_tissue)
            return new_tissues
        elif opt == 'write_deformation_fields':
            return super(MultiChannelNewSegment, self)._format_arg(opt, spec, [int(val[0]), int(val[1])])
        else:
            return super(MultiChannelNewSegment, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['native_class_images'] = []
        outputs['dartel_input_images'] = []
        outputs['normalized_class_images'] = []
        outputs['modulated_class_images'] = []
        outputs['transformation_mat'] = []
        outputs['bias_corrected_images'] = []
        outputs['bias_field_images'] = []
        outputs['inverse_deformation_field'] = []
        outputs['forward_deformation_field'] = []
        n_classes = 5
        if isdefined(self.inputs.tissues):
            n_classes = len(self.inputs.tissues)
        for i in range(n_classes):
            outputs['native_class_images'].append([])
            outputs['dartel_input_images'].append([])
            outputs['normalized_class_images'].append([])
            outputs['modulated_class_images'].append([])
        for filename in self.inputs.channels[0][0]:
            pth, base, ext = split_filename(filename)
            if isdefined(self.inputs.tissues):
                for i, tissue in enumerate(self.inputs.tissues):
                    if tissue[2][0]:
                        outputs['native_class_images'][i].append(os.path.join(pth, 'c%d%s.nii' % (i + 1, base)))
                    if tissue[2][1]:
                        outputs['dartel_input_images'][i].append(os.path.join(pth, 'rc%d%s.nii' % (i + 1, base)))
                    if tissue[3][0]:
                        outputs['normalized_class_images'][i].append(os.path.join(pth, 'wc%d%s.nii' % (i + 1, base)))
                    if tissue[3][1]:
                        outputs['modulated_class_images'][i].append(os.path.join(pth, 'mwc%d%s.nii' % (i + 1, base)))
            else:
                for i in range(n_classes):
                    outputs['native_class_images'][i].append(os.path.join(pth, 'c%d%s.nii' % (i + 1, base)))
            outputs['transformation_mat'].append(os.path.join(pth, '%s_seg8.mat' % base))
            if isdefined(self.inputs.write_deformation_fields):
                if self.inputs.write_deformation_fields[0]:
                    outputs['inverse_deformation_field'].append(os.path.join(pth, 'iy_%s.nii' % base))
                if self.inputs.write_deformation_fields[1]:
                    outputs['forward_deformation_field'].append(os.path.join(pth, 'y_%s.nii' % base))
        for channel in self.inputs.channels:
            for filename in channel[0]:
                pth, base, ext = split_filename(filename)
                if isdefined(channel[1]):
                    if channel[1][2][0]:
                        outputs['bias_field_images'].append(os.path.join(pth, 'BiasField_%s.nii' % base))
                    if channel[1][2][1]:
                        outputs['bias_corrected_images'].append(os.path.join(pth, 'm%s.nii' % base))
        return outputs