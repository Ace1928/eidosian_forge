import abc
import ast
import inspect
import stevedore
class BaseCheck(metaclass=abc.ABCMeta):
    """Abstract base class for Check classes."""
    scope_types = None

    @abc.abstractmethod
    def __str__(self):
        """String representation of the Check tree rooted at this node."""
        pass

    @abc.abstractmethod
    def __call__(self, target, cred, enforcer, current_rule=None):
        """Triggers if instance of the class is called.

        Performs the check. Returns False to reject the access or a
        true value (not necessary True) to accept the access.
        """
        pass