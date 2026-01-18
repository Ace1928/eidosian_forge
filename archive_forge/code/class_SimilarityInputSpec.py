import warnings
import numpy as np
import nibabel as nb
from .base import NipyBaseInterface, have_nipy
from ..base import TraitedSpec, traits, BaseInterfaceInputSpec, File, isdefined
class SimilarityInputSpec(BaseInterfaceInputSpec):
    volume1 = File(exists=True, desc='3D volume', mandatory=True)
    volume2 = File(exists=True, desc='3D volume', mandatory=True)
    mask1 = File(exists=True, desc='3D volume')
    mask2 = File(exists=True, desc='3D volume')
    metric = traits.Either(traits.Enum('cc', 'cr', 'crl1', 'mi', 'nmi', 'slr'), traits.Callable(), desc="str or callable\nCost-function for assessing image similarity. If a string,\none of 'cc': correlation coefficient, 'cr': correlation\nratio, 'crl1': L1-norm based correlation ratio, 'mi': mutual\ninformation, 'nmi': normalized mutual information, 'slr':\nsupervised log-likelihood ratio. If a callable, it should\ntake a two-dimensional array representing the image joint\nhistogram as an input and return a float.", usedefault=True)