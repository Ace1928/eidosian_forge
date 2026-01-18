import abc
import ast
import inspect
import stevedore
@register('role')
class RoleCheck(Check):
    """Check that there is a matching role in the ``creds`` dict."""

    def __call__(self, target, creds, enforcer, current_rule=None):
        try:
            match = self.match % target
        except KeyError:
            return False
        if 'roles' in creds:
            return match.lower() in [x.lower() for x in creds['roles']]
        return False