import functools
import inspect
import wrapt
from debtcollector import _utils
class moved_read_only_property(object):
    """Descriptor for read-only properties moved to another location.

    This works like the ``@property`` descriptor but can be used instead to
    provide the same functionality and also interact with the :mod:`warnings`
    module to warn when a property is accessed, so that users of those
    properties can know that a previously read-only property at a prior
    location/name has moved to another location/name.

    :param old_name: old attribute location/name
    :param new_name: new attribute location/name
    :param version: version string (represents the version this deprecation
                    was created in)
    :param removal_version: version string (represents the version this
                            deprecation will be removed in); a string
                            of '?' will denote this will be removed in
                            some future unknown version
    :param stacklevel: stacklevel used in the :func:`warnings.warn` function
                       to locate where the users code is when reporting the
                       deprecation call (the default being 3)
    :param category: the :mod:`warnings` category to use, defaults to
                     :py:class:`DeprecationWarning` if not provided
    """

    def __init__(self, old_name, new_name, version=None, removal_version=None, stacklevel=3, category=None):
        self._old_name = old_name
        self._new_name = new_name
        self._message = _utils.generate_message("Read-only property '%s' has moved to '%s'" % (self._old_name, self._new_name), version=version, removal_version=removal_version)
        self._stacklevel = stacklevel
        self._category = category

    def __get__(self, instance, owner):
        _utils.deprecation(self._message, stacklevel=self._stacklevel, category=self._category)
        if instance is not None:
            real_owner = instance
        else:
            real_owner = owner
        return getattr(real_owner, self._new_name)