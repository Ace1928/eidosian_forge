import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class TractSkeleton(FSLCommand):
    """Use FSL's tbss_skeleton to skeletonise an FA image or project arbitrary
    values onto a skeleton.

    There are two ways to use this interface.  To create a skeleton from an FA
    image, just supply the ``in_file`` and set ``skeleton_file`` to True (or
    specify a skeleton filename. To project values onto a skeleton, you must
    set ``project_data`` to True, and then also supply values for
    ``threshold``, ``distance_map``, and ``data_file``. The
    ``search_mask_file`` and ``use_cingulum_mask`` inputs are also used in data
    projection, but ``use_cingulum_mask`` is set to True by default.  This mask
    controls where the projection algorithm searches within a circular space
    around a tract, rather than in a single perpendicular direction.

    Example
    -------

    >>> import nipype.interfaces.fsl as fsl
    >>> skeletor = fsl.TractSkeleton()
    >>> skeletor.inputs.in_file = "all_FA.nii.gz"
    >>> skeletor.inputs.skeleton_file = True
    >>> skeletor.run() # doctest: +SKIP

    """
    _cmd = 'tbss_skeleton'
    input_spec = TractSkeletonInputSpec
    output_spec = TractSkeletonOutputSpec

    def _format_arg(self, name, spec, value):
        if name == 'project_data':
            if isdefined(value) and value:
                _si = self.inputs
                if isdefined(_si.use_cingulum_mask) and _si.use_cingulum_mask:
                    mask_file = Info.standard_image('LowerCingulum_1mm.nii.gz')
                else:
                    mask_file = _si.search_mask_file
                if not isdefined(_si.projected_data):
                    proj_file = self._list_outputs()['projected_data']
                else:
                    proj_file = _si.projected_data
                return spec.argstr % (_si.threshold, _si.distance_map, mask_file, _si.data_file, proj_file)
        elif name == 'skeleton_file':
            if isinstance(value, bool):
                return spec.argstr % self._list_outputs()['skeleton_file']
            else:
                return spec.argstr % value
        return super(TractSkeleton, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        _si = self.inputs
        if isdefined(_si.project_data) and _si.project_data:
            proj_data = _si.projected_data
            outputs['projected_data'] = proj_data
            if not isdefined(proj_data):
                stem = _si.data_file
                if isdefined(_si.alt_data_file):
                    stem = _si.alt_data_file
                outputs['projected_data'] = fname_presuffix(stem, suffix='_skeletonised', newpath=os.getcwd(), use_ext=True)
        if isdefined(_si.skeleton_file) and _si.skeleton_file:
            outputs['skeleton_file'] = _si.skeleton_file
            if isinstance(_si.skeleton_file, bool):
                outputs['skeleton_file'] = fname_presuffix(_si.in_file, suffix='_skeleton', newpath=os.getcwd(), use_ext=True)
        return outputs