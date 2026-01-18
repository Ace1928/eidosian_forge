import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class DARTELOutputSpec(TraitedSpec):
    final_template_file = File(exists=True, desc='final DARTEL template')
    template_files = traits.List(File(exists=True), desc='Templates from different stages of iteration')
    dartel_flow_fields = traits.List(File(exists=True), desc='DARTEL flow fields')