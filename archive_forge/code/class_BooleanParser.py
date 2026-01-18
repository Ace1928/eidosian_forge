from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import csv
import io
import string
from absl.flags import _helpers
import six
class BooleanParser(ArgumentParser):
    """Parser of boolean values."""

    def parse(self, argument):
        """See base class."""
        if isinstance(argument, six.string_types):
            if argument.lower() in ('true', 't', '1'):
                return True
            elif argument.lower() in ('false', 'f', '0'):
                return False
            else:
                raise ValueError('Non-boolean argument to boolean flag', argument)
        elif isinstance(argument, six.integer_types):
            bool_value = bool(argument)
            if argument == bool_value:
                return bool_value
            else:
                raise ValueError('Non-boolean argument to boolean flag', argument)
        raise TypeError('Non-boolean argument to boolean flag', argument)

    def flag_type(self):
        """See base class."""
        return 'bool'