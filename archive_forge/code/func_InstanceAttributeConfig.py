from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container import api_adapter as container_api_adapter
from googlecloudsdk.api_lib.krmapihosting import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import log
def InstanceAttributeConfig():
    return concepts.ResourceParameterAttributeConfig(name='name', help_text='The name of the Config Controller instance.')