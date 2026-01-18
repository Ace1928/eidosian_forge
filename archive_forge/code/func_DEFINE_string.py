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
def DEFINE_string(name, default, help, flag_values=_flagvalues.FLAGS, required=False, **args):
    """Registers a flag whose value can be any string."""
    parser = _argument_parser.ArgumentParser()
    serializer = _argument_parser.ArgumentSerializer()
    return DEFINE(parser, name, default, help, flag_values, serializer, required=required, **args)