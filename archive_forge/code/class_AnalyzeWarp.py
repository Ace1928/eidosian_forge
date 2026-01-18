import os.path as op
import re
from ... import logging
from .base import ElastixBaseInputSpec
from ..base import CommandLine, TraitedSpec, File, traits, InputMultiPath
class AnalyzeWarp(ApplyWarp):
    """
    Use transformix to get details from the input transform (generate
    the corresponding deformation field, generate the determinant of the
    Jacobian map or the Jacobian map itself)

    Example
    -------

    >>> from nipype.interfaces.elastix import AnalyzeWarp
    >>> reg = AnalyzeWarp()
    >>> reg.inputs.transform_file = 'TransformParameters.0.txt'
    >>> reg.cmdline
    'transformix -def all -jac all -jacmat all -threads 1 -out ./ -tp TransformParameters.0.txt'


    """
    input_spec = AnalyzeWarpInputSpec
    output_spec = AnalyzeWarpOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        out_dir = op.abspath(self.inputs.output_path)
        outputs['disp_field'] = op.join(out_dir, 'deformationField.nii.gz')
        outputs['jacdet_map'] = op.join(out_dir, 'spatialJacobian.nii.gz')
        outputs['jacmat_map'] = op.join(out_dir, 'fullSpatialJacobian.nii.gz')
        return outputs