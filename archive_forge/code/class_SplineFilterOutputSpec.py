import os
from ..base import (
class SplineFilterOutputSpec(TraitedSpec):
    smoothed_track_file = File(exists=True)