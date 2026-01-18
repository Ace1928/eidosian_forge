import os
from ... import logging
from ..base import TraitedSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
from .base import FSCommand, FSTraitedSpec, FSCommandOpenMP, FSTraitedSpecOpenMP
class FuseSegmentations(FSCommand):
    """fuse segmentations together from multiple timepoints

    Examples
    --------
    >>> from nipype.interfaces.freesurfer import FuseSegmentations
    >>> fuse = FuseSegmentations()
    >>> fuse.inputs.subject_id = 'tp.long.A.template'
    >>> fuse.inputs.timepoints = ['tp1', 'tp2']
    >>> fuse.inputs.out_file = 'aseg.fused.mgz'
    >>> fuse.inputs.in_segmentations = ['aseg.mgz', 'aseg.mgz']
    >>> fuse.inputs.in_segmentations_noCC = ['aseg.mgz', 'aseg.mgz']
    >>> fuse.inputs.in_norms = ['norm.mgz', 'norm.mgz', 'norm.mgz']
    >>> fuse.cmdline
    'mri_fuse_segmentations -n norm.mgz -a aseg.mgz -c aseg.mgz tp.long.A.template tp1 tp2'
    """
    _cmd = 'mri_fuse_segmentations'
    input_spec = FuseSegmentationsInputSpec
    output_spec = FuseSegmentationsOutputSpec

    def _format_arg(self, name, spec, value):
        if name in ('in_segmentations', 'in_segmentations_noCC', 'in_norms'):
            return spec.argstr % os.path.basename(value[0])
        return super(FuseSegmentations, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs