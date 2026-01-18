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
def AsObjectReference(self):
    return self._messages.ObjectReference(kind=self.kind, namespace=self.namespace, name=self.name, uid=self.uid, apiVersion=self.apiVersion)