from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddLocationFlag(parser, verb):
    parser.add_argument('--location', help='The location of the workforce pool{0} to {1}.'.format('s' if verb == 'list' else '', verb), required=True)