from holoviews.core import Element
from holoviews.operation.element import contours
from holoviews.operation.stats import bivariate_kde
from .. import element as gv_element
from ..element import _Element
from .projection import ( # noqa (API import)
from .resample import resample_geometry # noqa (API import)
def add_crs(op, element, **kwargs):
    """
    Converts any elements in the input to their equivalent geotypes
    if given a coordinate reference system.
    """
    return element.map(lambda x: convert_to_geotype(x, kwargs.get('crs')), Element)