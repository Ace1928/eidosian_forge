import abc
class FormattableColumn(object, metaclass=abc.ABCMeta):

    def __init__(self, value):
        self._value = value

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self._value == other._value

    def __lt__(self, other):
        return self.__class__ == other.__class__ and self._value < other._value

    def __str__(self):
        return self.human_readable()

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.machine_readable())

    @abc.abstractmethod
    def human_readable(self):
        """Return a basic human readable version of the data."""

    def machine_readable(self):
        """Return a raw data structure using only Python built-in types.

        It must be possible to serialize the return value directly
        using a formatter like JSON, and it will be up to the
        formatter plugin to decide how to make that transformation.
        """
        return self._value