import os
from ..base import (
class TrackMergeOutputSpec(TraitedSpec):
    track_file = File(exists=True)