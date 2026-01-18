import os
import os.path as op
import nibabel as nb
import numpy as np
from .. import config, logging
from ..interfaces.base import (
from ..interfaces.nipy.base import NipyBaseInterface
class FuzzyOverlapOutputSpec(TraitedSpec):
    jaccard = traits.Float(desc='Fuzzy Jaccard Index (fJI), all the classes')
    dice = traits.Float(desc='Fuzzy Dice Index (fDI), all the classes')
    class_fji = traits.List(traits.Float(), desc='Array containing the fJIs of each computed class')
    class_fdi = traits.List(traits.Float(), desc='Array containing the fDIs of each computed class')