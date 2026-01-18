import os.path as op
import re
from ... import logging
from .base import ElastixBaseInputSpec
from ..base import CommandLine, TraitedSpec, File, traits, InputMultiPath
class AnalyzeWarpOutputSpec(TraitedSpec):
    disp_field = File(desc='displacements field')
    jacdet_map = File(desc='det(Jacobian) map')
    jacmat_map = File(desc='Jacobian matrix map')