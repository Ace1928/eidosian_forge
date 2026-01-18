import os
import os.path
import posixpath
import stat
from .compat import Collection, Iterable, string_types, unicode
class AlreadyRegisteredError(Exception):
    """
	The :exc:`AlreadyRegisteredError` exception is raised when a pattern
	factory is registered under a name already in use.
	"""

    def __init__(self, name, pattern_factory):
        """
		Initializes the :exc:`AlreadyRegisteredError` instance.

		*name* (:class:`str`) is the name of the registered pattern.

		*pattern_factory* (:class:`~collections.abc.Callable`) is the
		registered pattern factory.
		"""
        super(AlreadyRegisteredError, self).__init__(name, pattern_factory)

    @property
    def message(self):
        """
		*message* (:class:`str`) is the error message.
		"""
        return '{name!r} is already registered for pattern factory:{pattern_factory!r}.'.format(name=self.name, pattern_factory=self.pattern_factory)

    @property
    def name(self):
        """
		*name* (:class:`str`) is the name of the registered pattern.
		"""
        return self.args[0]

    @property
    def pattern_factory(self):
        """
		*pattern_factory* (:class:`~collections.abc.Callable`) is the
		registered pattern factory.
		"""
        return self.args[1]