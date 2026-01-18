from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
def GetSerialConsoleLog(compute_client, compute_message, instance, project, zone):
    req = compute_message.ComputeInstancesGetSerialPortOutputRequest(instance=instance, project=project, port=1, start=0, zone=zone)
    return compute_client.instances.GetSerialPortOutput(req).contents