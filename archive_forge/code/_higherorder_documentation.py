import types
from ._impl import (
Create a ``MatchesPredicateWithParams`` matcher.

        :param predicate: A function that takes an object to match and
            additional params as given in ``*args`` and ``**kwargs``. The
            result of the function will be interpreted as a boolean to
            determine a match.
        :param message: A message to describe a mismatch.  It will be formatted
            with .format() and be given a tuple containing whatever was passed
            to ``match()`` + ``*args`` in ``*args``, and whatever was passed to
            ``**kwargs`` as its ``**kwargs``.

            For instance, to format a single parameter::

                "{0} is not a {1}"

            To format a keyword arg::

                "{0} is not a {type_to_check}"
        :param name: What name to use for the matcher class. Pass None to use
            the default.
        