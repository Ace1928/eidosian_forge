import collections
from absl import logging
class OptionsBase:
    """Base class for representing a set of tf.data options.

  Attributes:
    _options: Stores the option values.
  """

    def __init__(self):
        object.__setattr__(self, '_options', {})
        object.__setattr__(self, '_mutable', True)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        for name in set(self._options) | set(other._options):
            if getattr(self, name) != getattr(other, name):
                return False
        return True

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        else:
            return NotImplemented

    def __setattr__(self, name, value):
        if not self._mutable:
            raise ValueError('Mutating `tf.data.Options()` returned by `tf.data.Dataset.options()` has no effect. Use `tf.data.Dataset.with_options(options)` to set or update dataset options.')
        if hasattr(self, name):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError('Cannot set the property {} on {}.'.format(name, type(self).__name__))

    def _set_mutable(self, mutable):
        """Change the mutability property to `mutable`."""
        object.__setattr__(self, '_mutable', mutable)

    def _to_proto(self):
        """Convert options to protocol buffer."""
        raise NotImplementedError('{}._to_proto()'.format(type(self).__name__))

    def _from_proto(self, pb):
        """Convert protocol buffer to options."""
        raise NotImplementedError('{}._from_proto()'.format(type(self).__name__))