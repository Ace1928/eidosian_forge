from types import FunctionType, MethodType
import warnings
from .constants import (
from .ctrait import CTrait
from .trait_errors import TraitError
from .trait_base import (
from .trait_converters import (
from .trait_handler import TraitHandler
from .trait_type import (
from .trait_handlers import (
from .trait_factory import (
from .util.deprecated import deprecated
@deprecated("'RGBColor' in 'traits' package has been deprecated. Use 'RGBColor' from 'traitsui' package instead.")
def RGBColor(*args, **metadata):
    """ Returns a trait whose value must be a GUI toolkit-specific RGB-based
    color.

    .. deprecated:: 6.1.0
        ``RGBColor`` trait in this package will be removed in the future. It is
        replaced by ``RGBColor`` trait in TraitsUI package.
    """
    from traitsui.toolkit_traits import RGBColorTrait
    return RGBColorTrait(*args, **metadata)