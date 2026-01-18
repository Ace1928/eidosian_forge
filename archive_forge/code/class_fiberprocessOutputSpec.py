import os
from ....base import (
class fiberprocessOutputSpec(TraitedSpec):
    fiber_output = File(desc='Output fiber file. May be warped or updated with new data depending on other options used.', exists=True)
    voxelize = File(desc='Voxelize fiber into a label map (the labelmap filename is the argument of -V). The tensor file must be specified using -T for information about the size, origin, spacing of the image. The deformation is applied before the voxelization ', exists=True)