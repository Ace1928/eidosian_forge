import warnings
import numpy as np
import nibabel as nb
from .base import NipyBaseInterface, have_nipy
from ..base import TraitedSpec, traits, BaseInterfaceInputSpec, File, isdefined
class SimilarityOutputSpec(TraitedSpec):
    similarity = traits.Float(desc='Similarity between volume 1 and 2')