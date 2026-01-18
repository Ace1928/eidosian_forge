from .colorbrewer import (  # noqa: F401
from .cmocean import (  # noqa: F401
from .carto import (  # noqa: F401
from .plotlyjs import Picnic, Portland, Picnic_r, Portland_r  # noqa: F401
from ._swatches import _swatches, _swatches_continuous

Diverging color scales are appropriate for continuous data that has a natural midpoint other otherwise informative special value, such as 0 altitude, or the boiling point
of a liquid. The color scales in this module are mostly meant to be passed in as the `color_continuous_scale` argument to various functions, and to be used with the `color_continuous_midpoint` argument.
