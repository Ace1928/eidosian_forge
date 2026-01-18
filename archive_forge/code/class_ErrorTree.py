from collections import defaultdict, deque
import itertools
import pprint
import textwrap
from jsonschema import _utils
from jsonschema.compat import PY3, iteritems
class ErrorTree(object):
    """
    ErrorTrees make it easier to check which validations failed.

    """
    _instance = _unset

    def __init__(self, errors=()):
        self.errors = {}
        self._contents = defaultdict(self.__class__)
        for error in errors:
            container = self
            for element in error.path:
                container = container[element]
            container.errors[error.validator] = error
            container._instance = error.instance

    def __contains__(self, index):
        """
        Check whether ``instance[index]`` has any errors.

        """
        return index in self._contents

    def __getitem__(self, index):
        """
        Retrieve the child tree one level down at the given ``index``.

        If the index is not in the instance that this tree corresponds to and
        is not known by this tree, whatever error would be raised by
        ``instance.__getitem__`` will be propagated (usually this is some
        subclass of :class:`LookupError`.

        """
        if self._instance is not _unset and index not in self:
            self._instance[index]
        return self._contents[index]

    def __setitem__(self, index, value):
        self._contents[index] = value

    def __iter__(self):
        """
        Iterate (non-recursively) over the indices in the instance with errors.

        """
        return iter(self._contents)

    def __len__(self):
        """
        Same as :attr:`total_errors`.

        """
        return self.total_errors

    def __repr__(self):
        return '<%s (%s total errors)>' % (self.__class__.__name__, len(self))

    @property
    def total_errors(self):
        """
        The total number of errors in the entire tree, including children.

        """
        child_errors = sum((len(tree) for _, tree in iteritems(self._contents)))
        return len(self.errors) + child_errors