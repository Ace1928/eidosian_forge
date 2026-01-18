from six import PY2
from functools import wraps
from datetime import datetime, timedelta, tzinfo
class _DatetimeWithFold(datetime):
    """
        This is a class designed to provide a PEP 495-compliant interface for
        Python versions before 3.6. It is used only for dates in a fold, so
        the ``fold`` attribute is fixed at ``1``.

        .. versionadded:: 2.6.0
        """
    __slots__ = ()

    def replace(self, *args, **kwargs):
        """
            Return a datetime with the same attributes, except for those
            attributes given new values by whichever keyword arguments are
            specified. Note that tzinfo=None can be specified to create a naive
            datetime from an aware datetime with no conversion of date and time
            data.

            This is reimplemented in ``_DatetimeWithFold`` because pypy3 will
            return a ``datetime.datetime`` even if ``fold`` is unchanged.
            """
        argnames = ('year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond', 'tzinfo')
        for arg, argname in zip(args, argnames):
            if argname in kwargs:
                raise TypeError('Duplicate argument: {}'.format(argname))
            kwargs[argname] = arg
        for argname in argnames:
            if argname not in kwargs:
                kwargs[argname] = getattr(self, argname)
        dt_class = self.__class__ if kwargs.get('fold', 1) else datetime
        return dt_class(**kwargs)

    @property
    def fold(self):
        return 1