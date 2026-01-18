import functools
import types
from fixtures import Fixture
class MonkeyPatch(Fixture):
    """Replace or delete an attribute."""
    delete = object()

    def __init__(self, name, new_value=None):
        """Create a MonkeyPatch.

        :param name: The fully qualified object name to override.
        :param new_value: A value to set the name to. If set to
            MonkeyPatch.delete the attribute will be deleted.

        During setup the name will be deleted or assigned the requested value,
        and this will be restored in cleanUp.

        When patching methods, the call signature of name should be a subset
        of the parameters which can be used to call new_value.

        For instance.

        >>> class T:
        ...     def method(self, arg1):
        ...         pass
        >>> class N:
        ...     @staticmethod
        ...     def newmethod(arg1):
        ...         pass

        Patching N.newmethod on top of T.method and then calling T().method(1)
        will not work because they do not have compatible call signatures -
        self will be passed to newmethod because the callable (N.newmethod)
        is placed onto T as a regular function. This allows capturing all the
        supplied parameters while still consulting local state in your
        new_value.
        """
        Fixture.__init__(self)
        self.name = name
        self.new_value = new_value

    def _setUp(self):
        location, attribute = self.name.rsplit('.', 1)
        try:
            __import__(location, {}, {})
        except ImportError:
            pass
        components = location.split('.')
        current = __import__(components[0], {}, {})
        for component in components[1:]:
            current = getattr(current, component)
        sentinel = object()
        new_value, old_value = _coerce_values(current, attribute, self.new_value, sentinel)
        if self.new_value is self.delete:
            if old_value is not sentinel:
                delattr(current, attribute)
        else:
            setattr(current, attribute, new_value)
        if old_value is sentinel:
            self.addCleanup(self._safe_delete, current, attribute)
        else:
            self.addCleanup(setattr, current, attribute, old_value)

    def _safe_delete(self, obj, attribute):
        """Delete obj.attribute handling the case where its missing."""
        sentinel = object()
        if getattr(obj, attribute, sentinel) is not sentinel:
            delattr(obj, attribute)