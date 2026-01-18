import os
from .base import NipyBaseInterface
from ..base import (
class FitGLMOutputSpec(TraitedSpec):
    beta = File(exists=True)
    nvbeta = traits.Any()
    s2 = File(exists=True)
    dof = traits.Any()
    constants = traits.Any()
    axis = traits.Any()
    reg_names = traits.List()
    residuals = File()
    a = File(exists=True)