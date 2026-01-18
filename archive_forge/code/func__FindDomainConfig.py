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
def _FindDomainConfig(self, subresources: List[runapps_v1alpha1_messages.Resource], domain) -> Optional[runapps_v1alpha1_messages.Resource]:
    for res in subresources:
        if res.id.type == DOMAIN_TYPE:
            cfg = encoding.MessageToDict(res.config)
            if cfg.get('domain') == domain:
                return res
    return None