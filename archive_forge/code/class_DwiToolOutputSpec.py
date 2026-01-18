from ..base import File, TraitedSpec, traits, isdefined, CommandLineInputSpec
from .base import NiftyFitCommand
from ..niftyreg.base import get_custom_path
class DwiToolOutputSpec(TraitedSpec):
    """Output Spec for DwiTool."""
    desc = 'Filename of multi-compartment model parameter map (-ivim,-ball,-nod)'
    mcmap_file = File(desc=desc)
    syn_file = File(desc='Filename of synthetic image')
    mdmap_file = File(desc='Filename of MD map/ADC')
    famap_file = File(desc='Filename of FA map')
    v1map_file = File(desc='Filename of PDD map [x,y,z]')
    rgbmap_file = File(desc='Filename of colour FA map')
    logdti_file = File(desc='Filename of output logdti map')