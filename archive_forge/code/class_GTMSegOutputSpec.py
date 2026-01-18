import os
from ... import logging
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .model import GLMFitInputSpec, GLMFit
class GTMSegOutputSpec(TraitedSpec):
    out_file = File(desc='GTM segmentation')