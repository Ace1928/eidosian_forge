import os
import os.path
import posixpath
import stat
from .compat import Collection, Iterable, string_types, unicode
class RecursionError(Exception):
    """
	The :exc:`RecursionError` exception is raised when recursion is
	detected.
	"""

    def __init__(self, real_path, first_path, second_path):
        """
		Initializes the :exc:`RecursionError` instance.

		*real_path* (:class:`str`) is the real path that recursion was
		encountered on.

		*first_path* (:class:`str`) is the first path encountered for
		*real_path*.

		*second_path* (:class:`str`) is the second path encountered for
		*real_path*.
		"""
        super(RecursionError, self).__init__(real_path, first_path, second_path)

    @property
    def first_path(self):
        """
		*first_path* (:class:`str`) is the first path encountered for
		:attr:`self.real_path <RecursionError.real_path>`.
		"""
        return self.args[1]

    @property
    def message(self):
        """
		*message* (:class:`str`) is the error message.
		"""
        return 'Real path {real!r} was encountered at {first!r} and then {second!r}.'.format(real=self.real_path, first=self.first_path, second=self.second_path)

    @property
    def real_path(self):
        """
		*real_path* (:class:`str`) is the real path that recursion was
		encountered on.
		"""
        return self.args[0]

    @property
    def second_path(self):
        """
		*second_path* (:class:`str`) is the second path encountered for
		:attr:`self.real_path <RecursionError.real_path>`.
		"""
        return self.args[2]