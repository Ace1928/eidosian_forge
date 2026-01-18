import warnings
from .constants import ComparisonMode, DefaultValue
from .trait_base import SequenceTypes
from .trait_errors import TraitError
from .trait_type import TraitType
from .trait_types import Str, Any, Int as TInt, Float as TFloat
class ArrayOrNone(CArray):
    """ A coercing trait whose value may be either a NumPy array or None.

    This trait is designed to avoid the comparison issues with numpy arrays
    that can arise from the use of constructs like Either(None, Array).

    The default value is None.
    """

    def __init__(self, *args, **metadata):
        metadata.setdefault('comparison_mode', ComparisonMode.identity)
        super().__init__(*args, **metadata)

    def validate(self, object, name, value):
        if value is None:
            return value
        return super().validate(object, name, value)

    def get_default_value(self):
        dv = self.default_value
        if dv is None:
            return (DefaultValue.constant, dv)
        else:
            return (DefaultValue.callable_and_args, (self.copy_default_value, (self.validate(None, None, dv),), None))

    def _default_for_dtype_and_shape(self, dtype, shape):
        return None