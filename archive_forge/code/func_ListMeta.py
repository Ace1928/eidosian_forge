from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.run import condition
from googlecloudsdk.core.console import console_attr
import six
def ListMeta(m):
    """List Metadta class from messages module."""
    if hasattr(m, 'ListMeta'):
        return m.ListMeta
    elif hasattr(m, 'K8sIoApimachineryPkgApisMetaV1ListMeta'):
        return m.K8sIoApimachineryPkgApisMetaV1ListMeta
    raise ValueError('Provided module does not have a known metadata class')