from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import List
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.command_lib.run.integrations.typekits import base
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages as runapps
def _CountType(counter, res_id):
    counter[res_id.type] = counter.get(res_id.type, 0) + 1