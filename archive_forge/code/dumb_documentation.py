import ast as _ast
import io as _io
import os as _os
import collections.abc
Open the database file, filename, and return corresponding object.

    The flag argument, used to control how the database is opened in the
    other DBM implementations, supports only the semantics of 'c' and 'n'
    values.  Other values will default to the semantics of 'c' value:
    the database will always opened for update and will be created if it
    does not exist.

    The optional mode argument is the UNIX mode of the file, used only when
    the database has to be created.  It defaults to octal code 0o666 (and
    will be modified by the prevailing umask).

    