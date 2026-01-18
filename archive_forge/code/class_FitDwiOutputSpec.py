from ..base import File, TraitedSpec, traits, isdefined, CommandLineInputSpec
from .base import NiftyFitCommand
from ..niftyreg.base import get_custom_path
class FitDwiOutputSpec(TraitedSpec):
    """Output Spec for FitDwi."""
    error_file = File(desc='Filename of parameter error maps')
    res_file = File(desc='Filename of model residual map')
    syn_file = File(desc='Filename of synthetic image')
    nodiff_file = File(desc='Filename of average no diffusion image.')
    mdmap_file = File(desc='Filename of MD map/ADC')
    famap_file = File(desc='Filename of FA map')
    v1map_file = File(desc='Filename of PDD map [x,y,z]')
    rgbmap_file = File(desc='Filename of colour FA map')
    tenmap_file = File(desc='Filename of tensor map')
    tenmap2_file = File(desc='Filename of tensor map [lower tri]')
    mcmap_file = File(desc='Filename of multi-compartment model parameter map (-ivim,-ball,-nod).')
    mcout = File(desc='Filename of mc samples (ascii text file)')