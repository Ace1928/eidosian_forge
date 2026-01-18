import os
from .base import (
class Bru2OutputSpec(TraitedSpec):
    nii_file = File(exists=True)