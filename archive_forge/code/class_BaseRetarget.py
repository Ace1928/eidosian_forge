import abc
import weakref
from numba.core import errors
class BaseRetarget(abc.ABC):
    """Abstract base class for retargeting logic.
    """

    @abc.abstractmethod
    def check_compatible(self, orig_disp):
        """Check that the retarget is compatible.

        This method does not return anything meaningful (e.g. None)
        Incompatibility is signalled via raising an exception.
        """
        pass

    @abc.abstractmethod
    def retarget(self, orig_disp):
        """Retargets the given dispatcher and returns a new dispatcher-like
        callable. Or, returns the original dispatcher if the the target_backend
        will not change.
        """
        pass