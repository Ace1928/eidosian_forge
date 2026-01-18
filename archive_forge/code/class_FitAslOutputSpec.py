from ..base import File, TraitedSpec, traits, CommandLineInputSpec
from .base import NiftyFitCommand
from ..niftyreg.base import get_custom_path
class FitAslOutputSpec(TraitedSpec):
    """Output Spec for FitAsl."""
    desc = 'Filename of the Cerebral Blood Flow map (in ml/100g/min).'
    cbf_file = File(exists=True, desc=desc)
    desc = 'Filename of the CBF error map.'
    error_file = File(exists=True, desc=desc)
    desc = 'Filename of the synthetic ASL data.'
    syn_file = File(exists=True, desc=desc)