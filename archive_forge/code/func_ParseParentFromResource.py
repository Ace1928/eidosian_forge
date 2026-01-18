from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import maps
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util import parameter_info_lib
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
def ParseParentFromResource(resource_ref):
    collection_list = resource_ref.Collection().split('.')
    parent_collection = '.'.join(collection_list[:-1])
    params = resource_ref.AsDict()
    del params[collection_list[-1] + 'Id']
    return resources.REGISTRY.Create(parent_collection, **params)