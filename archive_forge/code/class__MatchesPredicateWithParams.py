import types
from ._impl import (
class _MatchesPredicateWithParams(Matcher):

    def __init__(self, predicate, message, name, *args, **kwargs):
        """Create a ``MatchesPredicateWithParams`` matcher.

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
        """
        self.predicate = predicate
        self.message = message
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        args = [str(arg) for arg in self.args]
        kwargs = ['%s=%s' % item for item in self.kwargs.items()]
        args = ', '.join(args + kwargs)
        if self.name is None:
            name = 'MatchesPredicateWithParams({!r}, {!r})'.format(self.predicate, self.message)
        else:
            name = self.name
        return f'{name}({args})'

    def match(self, x):
        if not self.predicate(x, *self.args, **self.kwargs):
            return Mismatch(self.message.format(*(x,) + self.args, **self.kwargs))