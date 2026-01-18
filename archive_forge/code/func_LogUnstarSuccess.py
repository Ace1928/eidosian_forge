from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.data_catalog import entries
from googlecloudsdk.api_lib.data_catalog import util as api_util
from googlecloudsdk.command_lib.concepts import exceptions as concept_exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
import six
def LogUnstarSuccess(_, args):
    log.out.Print('Unstarred entry [{}].'.format(args.entry))