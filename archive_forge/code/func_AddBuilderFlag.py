from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddBuilderFlag(parser):
    parser.add_argument('--builder', help='Name of the builder to use for pack, e.g. ' + '`gcr.io/gae-runtimes/buildpacks/google-gae-22/go/builder`.')