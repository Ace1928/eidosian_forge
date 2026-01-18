from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict, List, Optional
from apitools.base.py import encoding
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.command_lib.run.integrations.typekits import base
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
def _ParseMappingNotation(self, mapping):
    mapping_parts = mapping.split(':')
    if len(mapping_parts) != 2:
        raise exceptions.ArgumentError('Mapping "{}" is not valid. Missing service notation.'.format(mapping))
    url = mapping_parts[0]
    service = mapping_parts[1]
    return (url, service)