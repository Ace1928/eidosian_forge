import os
import warnings
from enchant.errors import Error, DictNotFoundError
from enchant.utils import get_default_language
from enchant.pypwl import PyPWL
class _EnchantObject:
    """Base class for enchant objects.

    This class implements some general functionality for interfacing with
    the '_enchant' C-library in a consistent way.  All public objects
    from the 'enchant' module are subclasses of this class.

    All enchant objects have an attribute '_this' which contains the
    pointer to the underlying C-library object.  The method '_check_this'
    can be called to ensure that this point is not None, raising an
    exception if it is.
    """

    def __init__(self):
        """_EnchantObject constructor."""
        self._this = None
        if _e is not None:
            self._init_this()

    def _check_this(self, msg=None):
        """Check that self._this is set to a pointer, rather than None."""
        if self._this is None:
            if msg is None:
                msg = '%s unusable: the underlying C-library object has been freed.'
                msg = msg % (self.__class__.__name__,)
            raise Error(msg)

    def _init_this(self):
        """Initialise the underlying C-library object pointer."""
        raise NotImplementedError

    def _raise_error(self, default='Unspecified Error', eclass=Error):
        """Raise an exception based on available error messages.

        This method causes an Error to be raised.  Subclasses should
        override it to retrieve an error indication from the underlying
        API if possible.  If such a message cannot be retrieved, the
        argument value <default> is used.  The class of the exception
        can be specified using the argument <eclass>
        """
        raise eclass(default)
    _raise_error._DOC_ERRORS = ['eclass']

    def __getstate__(self):
        """Customize pickling of PyEnchant objects.

        Since it's not safe for multiple objects to share the same C-library
        object, we make sure it's unset when pickling.
        """
        state = self.__dict__.copy()
        state['_this'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_this()