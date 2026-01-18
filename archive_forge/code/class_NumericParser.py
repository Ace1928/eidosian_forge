from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import csv
import io
import string
from absl.flags import _helpers
import six
class NumericParser(ArgumentParser):
    """Parser of numeric values.

  Parsed value may be bounded to a given upper and lower bound.
  """

    def is_outside_bounds(self, val):
        """Returns whether the value is outside the bounds or not."""
        return self.lower_bound is not None and val < self.lower_bound or (self.upper_bound is not None and val > self.upper_bound)

    def parse(self, argument):
        """See base class."""
        val = self.convert(argument)
        if self.is_outside_bounds(val):
            raise ValueError('%s is not %s' % (val, self.syntactic_help))
        return val

    def _custom_xml_dom_elements(self, doc):
        elements = []
        if self.lower_bound is not None:
            elements.append(_helpers.create_xml_dom_element(doc, 'lower_bound', self.lower_bound))
        if self.upper_bound is not None:
            elements.append(_helpers.create_xml_dom_element(doc, 'upper_bound', self.upper_bound))
        return elements

    def convert(self, argument):
        """Returns the correct numeric value of argument.

    Subclass must implement this method, and raise TypeError if argument is not
    string or has the right numeric type.

    Args:
      argument: string argument passed in the commandline, or the numeric type.

    Raises:
      TypeError: Raised when argument is not a string or the right numeric type.
      ValueError: Raised when failed to convert argument to the numeric value.
    """
        raise NotImplementedError