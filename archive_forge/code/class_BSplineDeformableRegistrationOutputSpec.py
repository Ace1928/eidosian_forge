from nipype.interfaces.base import (
import os
class BSplineDeformableRegistrationOutputSpec(TraitedSpec):
    outputtransform = File(desc='Transform calculated that aligns the fixed and moving image. Maps positions from the fixed coordinate frame to the moving coordinate frame. Optional (specify an output transform or an output volume or both).', exists=True)
    outputwarp = File(desc='Vector field that applies an equivalent warp as the BSpline. Maps positions from the fixed coordinate frame to the moving coordinate frame. Optional.', exists=True)
    resampledmovingfilename = File(desc='Resampled moving image to fixed image coordinate frame. Optional (specify an output transform or an output volume or both).', exists=True)