import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class FSLXCommandOutputSpec(TraitedSpec):
    dyads = OutputMultiPath(File(exists=True), desc='Mean of PDD distribution in vector form.')
    fsamples = OutputMultiPath(File(exists=True), desc='Samples from the distribution on f anisotropy')
    mean_dsamples = File(exists=True, desc='Mean of distribution on diffusivity d')
    mean_fsamples = OutputMultiPath(File(exists=True), desc='Mean of distribution on f anisotropy')
    mean_S0samples = File(exists=True, desc='Mean of distribution on T2wbaseline signal intensity S0')
    mean_tausamples = File(exists=True, desc='Mean of distribution on tau samples (only with rician noise)')
    phsamples = OutputMultiPath(File(exists=True), desc='phi samples, per fiber')
    thsamples = OutputMultiPath(File(exists=True), desc='theta samples, per fiber')