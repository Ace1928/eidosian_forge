from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import types
from absl.flags import _argument_parser
from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _flagvalues
from absl.flags import _helpers
from absl.flags import _validators
class _FlagAlias(_flag.Flag):
    """Overrides Flag class so alias value is copy of original flag value."""

    def parse(self, argument):
        flag.parse(argument)
        self.present += 1

    def _parse_from_default(self, value):
        return value

    @property
    def value(self):
        return flag.value

    @value.setter
    def value(self, value):
        flag.value = value