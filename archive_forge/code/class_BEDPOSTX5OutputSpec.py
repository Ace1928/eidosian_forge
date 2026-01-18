import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class BEDPOSTX5OutputSpec(TraitedSpec):
    mean_dsamples = File(exists=True, desc='Mean of distribution on diffusivity d')
    mean_fsamples = OutputMultiPath(File(exists=True), desc='Mean of distribution on f anisotropy')
    mean_S0samples = File(exists=True, desc='Mean of distribution on T2wbaseline signal intensity S0')
    mean_phsamples = OutputMultiPath(File(exists=True), desc='Mean of distribution on phi')
    mean_thsamples = OutputMultiPath(File(exists=True), desc='Mean of distribution on theta')
    merged_thsamples = OutputMultiPath(File(exists=True), desc='Samples from the distribution on theta')
    merged_phsamples = OutputMultiPath(File(exists=True), desc='Samples from the distribution on phi')
    merged_fsamples = OutputMultiPath(File(exists=True), desc='Samples from the distribution on anisotropic volume fraction')
    dyads = OutputMultiPath(File(exists=True), desc='Mean of PDD distribution in vector form.')
    dyads_dispersion = OutputMultiPath(File(exists=True), desc='Dispersion')