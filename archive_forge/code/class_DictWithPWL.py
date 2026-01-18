import os
import warnings
from enchant.errors import Error, DictNotFoundError
from enchant.utils import get_default_language
from enchant.pypwl import PyPWL
class DictWithPWL(Dict):
    """Dictionary with separately-managed personal word list.

    .. note::
        As of version 1.4.0, enchant manages a per-user pwl and
        exclude list.  This class is now only needed if you want
        to explicitly maintain a separate word list in addition to
        the default one.

    This class behaves as the standard Dict class, but also manages a
    personal word list stored in a separate file.  The file must be
    specified at creation time by the 'pwl' argument to the constructor.
    Words added to the dictionary are automatically appended to the pwl file.

    A personal exclude list can also be managed, by passing another filename
    to the constructor in the optional 'pel' argument.  If this is not given,
    requests to exclude words are ignored.

    If either 'pwl' or 'pel' are None, an in-memory word list is used.
    This will prevent calls to add() and remove() from affecting the user's
    default word lists.

    The Dict object managing the PWL is available as the 'pwl' attribute.
    The Dict object managing the PEL is available as the 'pel' attribute.

    To create a DictWithPWL from the user's default language, use None
    as the 'tag' argument.
    """
    _DOC_ERRORS = ['pel', 'pel', 'PEL', 'pel']

    def __init__(self, tag, pwl=None, pel=None, broker=None):
        """DictWithPWL constructor.

        The argument 'pwl', if not None, names a file containing the
        personal word list.  If this file does not exist, it is created
        with default permissions.

        The argument 'pel', if not None, names a file containing the personal
        exclude list.  If this file does not exist, it is created with
        default permissions.
        """
        super().__init__(tag, broker)
        if pwl is not None:
            if not os.path.exists(pwl):
                f = open(pwl, 'wt')
                f.close()
                del f
            self.pwl = self._broker.request_pwl_dict(pwl)
        else:
            self.pwl = PyPWL()
        if pel is not None:
            if not os.path.exists(pel):
                f = open(pel, 'wt')
                f.close()
                del f
            self.pel = self._broker.request_pwl_dict(pel)
        else:
            self.pel = PyPWL()

    def _check_this(self, msg=None):
        """Extend Dict._check_this() to check PWL validity."""
        if self.pwl is None:
            self._free()
        if self.pel is None:
            self._free()
        super()._check_this(msg)
        self.pwl._check_this(msg)
        self.pel._check_this(msg)

    def _free(self):
        """Extend Dict._free() to free the PWL as well."""
        if self.pwl is not None:
            self.pwl._free()
            self.pwl = None
        if self.pel is not None:
            self.pel._free()
            self.pel = None
        super()._free()

    def check(self, word):
        """Check spelling of a word.

        This method takes a word in the dictionary language and returns
        True if it is correctly spelled, and false otherwise.  It checks
        both the dictionary and the personal word list.
        """
        if self.pel.check(word):
            return False
        if self.pwl.check(word):
            return True
        if super().check(word):
            return True
        return False

    def suggest(self, word):
        """Suggest possible spellings for a word.

        This method tries to guess the correct spelling for a given
        word, returning the possibilities in a list.
        """
        suggs = super().suggest(word)
        suggs.extend([w for w in self.pwl.suggest(word) if w not in suggs])
        for i in range(len(suggs) - 1, -1, -1):
            if self.pel.check(suggs[i]):
                del suggs[i]
        return suggs

    def add(self, word):
        """Add a word to the associated personal word list.

        This method adds the given word to the personal word list, and
        automatically saves the list to disk.
        """
        self._check_this()
        self.pwl.add(word)
        self.pel.remove(word)

    def remove(self, word):
        """Add a word to the associated exclude list."""
        self._check_this()
        self.pwl.remove(word)
        self.pel.add(word)

    def add_to_pwl(self, word):
        """Add a word to the associated personal word list.

        This method adds the given word to the personal word list, and
        automatically saves the list to disk.
        """
        self._check_this()
        self.pwl.add_to_pwl(word)
        self.pel.remove(word)

    def is_added(self, word):
        """Check whether a word is in the personal word list."""
        self._check_this()
        return self.pwl.is_added(word)

    def is_removed(self, word):
        """Check whether a word is in the personal exclude list."""
        self._check_this()
        return self.pel.is_added(word)