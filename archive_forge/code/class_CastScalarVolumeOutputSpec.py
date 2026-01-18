from nipype.interfaces.base import (
import os
class CastScalarVolumeOutputSpec(TraitedSpec):
    OutputVolume = File(position=-1, desc='Output volume, cast to the new type.', exists=True)