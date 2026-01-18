from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
import six
def GetClearLabelsFlag(labels_name='labels'):
    return base.Argument('--clear-{}'.format(labels_name), action='store_true', help="          Remove all labels. If `--update-{labels}` is also specified then\n          `--clear-{labels}` is applied first.\n\n          For example, to remove all labels:\n\n              $ {{command}} --clear-{labels}\n\n          To remove all existing labels and create two new labels,\n          ``foo'' and ``baz'':\n\n              $ {{command}} --clear-{labels} --update-{labels} foo=bar,baz=qux\n          ".format(labels=labels_name))