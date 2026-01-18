from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def ParseLocation(args):
    if not args.IsSpecified('location'):
        return 'locations/global'
    return 'locations/{}'.format(args.location)