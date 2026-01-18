from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
def ParseClearableField(args, arg_name):
    clear = getattr(args, 'clear_' + arg_name)
    set_ = getattr(args, arg_name, None)
    if clear:
        return UpdateResult.MakeWithUpdate(None)
    elif set_:
        return UpdateResult.MakeWithUpdate(set_)
    else:
        return UpdateResult.MakeNoUpdate()