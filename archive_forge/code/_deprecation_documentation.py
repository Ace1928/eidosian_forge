import functools
import textwrap
import warnings
Marks functions as deprecated.

    It will result in a warning when the function is called and a note in the
    docstring.

    Args:
        since: The version when the function was first deprecated.
        removed_in: The version when the function will be removed.
        instructions: The action users should take.
    