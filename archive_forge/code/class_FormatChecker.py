import datetime
import re
import socket
from jsonschema.compat import str_types
from jsonschema.exceptions import FormatError
class FormatChecker(object):
    """
    A ``format`` property checker.

    JSON Schema does not mandate that the ``format`` property actually do any
    validation. If validation is desired however, instances of this class can
    be hooked into validators to enable format validation.

    :class:`FormatChecker` objects always return ``True`` when asked about
    formats that they do not know how to validate.

    To check a custom format using a function that takes an instance and
    returns a ``bool``, use the :meth:`FormatChecker.checks` or
    :meth:`FormatChecker.cls_checks` decorators.

    Arguments:

        formats (iterable):

            The known formats to validate. This argument can be used to
            limit which formats will be used during validation.

    """
    checkers = {}

    def __init__(self, formats=None):
        if formats is None:
            self.checkers = self.checkers.copy()
        else:
            self.checkers = dict(((k, self.checkers[k]) for k in formats))

    def checks(self, format, raises=()):
        """
        Register a decorated function as validating a new format.

        Arguments:

            format (str):

                The format that the decorated function will check.

            raises (Exception):

                The exception(s) raised by the decorated function when
                an invalid instance is found.

                The exception object will be accessible as the
                :attr:`ValidationError.cause` attribute of the resulting
                validation error.

        """

        def _checks(func):
            self.checkers[format] = (func, raises)
            return func
        return _checks
    cls_checks = classmethod(checks)

    def check(self, instance, format):
        """
        Check whether the instance conforms to the given format.

        Arguments:

            instance (any primitive type, i.e. str, number, bool):

                The instance to check

            format (str):

                The format that instance should conform to


        Raises:

            :exc:`FormatError` if instance does not conform to ``format``

        """
        if format not in self.checkers:
            return
        func, raises = self.checkers[format]
        result, cause = (None, None)
        try:
            result = func(instance)
        except raises as e:
            cause = e
        if not result:
            raise FormatError('%r is not a %r' % (instance, format), cause=cause)

    def conforms(self, instance, format):
        """
        Check whether the instance conforms to the given format.

        Arguments:

            instance (any primitive type, i.e. str, number, bool):

                The instance to check

            format (str):

                The format that instance should conform to

        Returns:

            bool: Whether it conformed

        """
        try:
            self.check(instance, format)
        except FormatError:
            return False
        else:
            return True