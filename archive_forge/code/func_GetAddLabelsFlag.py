from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.api_lib.ml.vision import api_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.console import console_io
def GetAddLabelsFlag(labels_name='product-labels'):
    return base.Argument('--add-{}'.format(labels_name), metavar='KEY=VALUE', type=arg_parsers.ArgList(), action='append', help="          List of product labels to add. If a label already exists, it is\n          silently ignored.\n\n          For example, to add the product labels 'foo=baz' and 'baz=qux':\n\n              $ {{command}} --add-{labels}='foo=baz,baz=qux'\n\n          To change the product label 'foo=baz' to 'foo=qux':\n\n              $ {{command}} --remove-{labels}='foo=baz' --add-{labels}='foo-qux'\n          ".format(labels=labels_name))