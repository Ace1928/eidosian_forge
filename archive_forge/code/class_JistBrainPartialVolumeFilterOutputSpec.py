import os
from ..base import (
class JistBrainPartialVolumeFilterOutputSpec(TraitedSpec):
    outPartial = File(desc='Partial Volume Image', exists=True)