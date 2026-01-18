from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import functools
from absl._collections_abc import abc
from absl.flags import _argument_parser
from absl.flags import _exceptions
from absl.flags import _helpers
import six
class BooleanFlag(Flag):
    """Basic boolean flag.

  Boolean flags do not take any arguments, and their value is either
  True (1) or False (0).  The false value is specified on the command
  line by prepending the word 'no' to either the long or the short flag
  name.

  For example, if a Boolean flag was created whose long name was
  'update' and whose short name was 'x', then this flag could be
  explicitly unset through either --noupdate or --nox.
  """

    def __init__(self, name, default, help, short_name=None, **args):
        p = _argument_parser.BooleanParser()
        super(BooleanFlag, self).__init__(p, None, name, default, help, short_name, 1, **args)