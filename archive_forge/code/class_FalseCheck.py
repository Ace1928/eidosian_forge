import abc
import ast
import inspect
import stevedore
class FalseCheck(BaseCheck):
    """A policy check that always returns ``False`` (disallow)."""

    def __str__(self):
        """Return a string representation of this check."""
        return '!'

    def __call__(self, target, cred, enforcer, current_rule=None):
        """Check the policy."""
        return False