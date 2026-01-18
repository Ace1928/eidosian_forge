from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddAudienceArg(parser):
    parser.add_argument('--audiences', type=str, metavar='AUDIENCES', help='Intended recipient of the token. Currently, only one audience can be specified.')