from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.kuberun import revision
from googlecloudsdk.command_lib.kuberun import k8s_object_printer
from googlecloudsdk.command_lib.kuberun import kubernetes_consts
from googlecloudsdk.core.resource import custom_printer_base as cp
import six
def FormatSecretVolumeSource(v):
    if v.items:
        return '{}:{}'.format(v.secretName, v.items[0].key)
    else:
        return v.secretName