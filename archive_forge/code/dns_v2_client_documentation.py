from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dns.v2 import dns_v2_messages as messages
Updates an existing Response Policy Rule.

      Args:
        request: (DnsResponsePolicyRulesUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResponsePolicyRulesUpdateResponse) The response message.
      