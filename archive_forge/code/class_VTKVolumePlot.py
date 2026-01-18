from bokeh.core.enums import enumeration
from bokeh.core.has_props import abstract
from bokeh.core.properties import (
from bokeh.models import ColorMapper, Model
from ..config import config
from ..io.resources import bundled_files
from ..util import classproperty
from .layout import HTMLBox
class VTKVolumePlot(AbstractVTKPlot):
    """
    Bokeh model dedicated to plot a volumetric object with the help of vtk-js
    """
    ambient = Float(default=0.2)
    colormap = String(help='Colormap Name')
    controller_expanded = Bool(default=True, help='\n        If True the volume controller panel options is expanded in the view')
    data = Nullable(Dict(String, Any))
    diffuse = Float(default=0.7)
    display_slices = Bool(default=False)
    display_volume = Bool(default=True)
    edge_gradient = Float(default=0.2)
    interpolation = Enum(enumeration('fast_linear', 'linear', 'nearest'))
    mapper = Dict(String, Any)
    nan_opacity = Float(default=1)
    render_background = String(default='#52576e')
    rescale = Bool(default=False)
    sampling = Float(default=0.4)
    shadow = Bool(default=True)
    slice_i = Int(default=0)
    slice_j = Int(default=0)
    slice_k = Int(default=0)
    specular = Float(default=0.3)
    specular_power = Float(default=8.0)