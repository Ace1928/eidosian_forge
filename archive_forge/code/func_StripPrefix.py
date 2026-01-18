from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def StripPrefix(arg_name):
    if arg_name.startswith(PREFIX):
        return arg_name[len(PREFIX):]
    return arg_name