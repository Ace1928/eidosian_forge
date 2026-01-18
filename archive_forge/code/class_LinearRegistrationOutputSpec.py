from nipype.interfaces.base import (
import os
class LinearRegistrationOutputSpec(TraitedSpec):
    outputtransform = File(desc='Transform calculated that aligns the fixed and moving image. Maps positions in the fixed coordinate frame to the moving coordinate frame. Optional (specify an output transform or an output volume or both).', exists=True)
    resampledmovingfilename = File(desc='Resampled moving image to the fixed image coordinate frame. Optional (specify an output transform or an output volume or both).', exists=True)