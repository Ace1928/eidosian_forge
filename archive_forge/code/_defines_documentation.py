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
Overrides Flag class so alias value is copy of original flag value.