from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib import memcache
def ChooseUpdateMethod(unused_ref, args):
    if args.IsSpecified('parameters'):
        return 'updateParameters'
    return 'patch'