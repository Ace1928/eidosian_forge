from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def NamespaceFormat(arg_name):
    if IsPositional(arg_name):
        return arg_name
    return NormalizeFormat(arg_name)