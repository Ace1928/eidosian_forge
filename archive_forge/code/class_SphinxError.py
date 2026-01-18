from typing import Any, Optional
class SphinxError(Exception):
    """Base class for Sphinx errors.

    This is the base class for "nice" exceptions.  When such an exception is
    raised, Sphinx will abort the build and present the exception category and
    message to the user.

    Extensions are encouraged to derive from this exception for their custom
    errors.

    Exceptions *not* derived from :exc:`SphinxError` are treated as unexpected
    and shown to the user with a part of the traceback (and the full traceback
    saved in a temporary file).

    .. attribute:: category

       Description of the exception "category", used in converting the
       exception to a string ("category: message").  Should be set accordingly
       in subclasses.
    """
    category = 'Sphinx error'