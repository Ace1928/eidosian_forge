import warnings
import numpy as np
import param
from packaging.version import Version
from param import _is_number
from ..core import (
from ..core.data import ArrayInterface, DictInterface, PandasInterface, default_datatype
from ..core.data.util import dask_array_module
from ..core.util import (
from ..element.chart import Histogram, Scatter
from ..element.path import Contours, Polygons
from ..element.raster import RGB, Image
from ..element.util import categorical_aggregate2d  # noqa (API import)
from ..streams import RangeXY
from ..util.locator import MaxNLocator
class transform(Operation):
    """
    Generic Operation to transform an input Image or RGBA
    element into an output Image. The transformation is defined by
    the supplied callable that accepts the data of the input Image
    (typically a numpy array) and returns the transformed data of the
    output Image.

    This operator is extremely versatile; for instance, you could
    implement an alternative to the explicit threshold operator with:

    operator=lambda x: np.clip(x, 0, 0.5)

    Alternatively, you can implement a transform computing the 2D
    autocorrelation using the scipy library with:

    operator=lambda x: scipy.signal.correlate2d(x, x)
    """
    output_type = Image
    group = param.String(default='Transform', doc='\n        The group assigned to the result after applying the\n        transform.')
    operator = param.Callable(doc='\n       Function of one argument that transforms the data in the input\n       Image to the data in the output Image. By default, acts as\n       the identity function such that the output matches the input.')

    def _process(self, img, key=None):
        processed = img.data if not self.p.operator else self.p.operator(img.data)
        return img.clone(processed, group=self.p.group)