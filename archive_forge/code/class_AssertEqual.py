import os
import re
import numpy as np
from ..base import (
from ..io import IOBase, add_traits
from ...utils.filemanip import ensure_list, copyfile, split_filename
class AssertEqual(BaseInterface):
    input_spec = AssertEqualInputSpec

    def _run_interface(self, runtime):
        import nibabel as nb
        data1 = np.asanyarray(nb.load(self.inputs.volume1))
        data2 = np.asanyarray(nb.load(self.inputs.volume2))
        if not np.array_equal(data1, data2):
            raise RuntimeError('Input images are not exactly equal')
        return runtime