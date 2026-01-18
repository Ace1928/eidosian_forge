from ..base import (
import os
class ICA_AROMAOutputSpec(TraitedSpec):
    aggr_denoised_file = File(exists=True, desc='if generated: aggressively denoised volume')
    nonaggr_denoised_file = File(exists=True, desc='if generated: non aggressively denoised volume')
    out_dir = Directory(exists=True, desc='directory contains (in addition to the denoised files): melodic.ica + classified_motion_components + classification_overview + feature_scores + melodic_ic_mni)')