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
@classmethod
def SpecOnly(cls, spec, messages_mod, kind=None):
    """Produces a wrapped message with only the given spec.

    It is meant to be used as part of another message; it will error if you
    try to access the metadata or status.

    Arguments:
      spec: messages.Message, The spec to include
      messages_mod: the messages module
      kind: str, the resource kind

    Returns:
      A new k8s_object with only the given spec.
    """
    msg_cls = getattr(messages_mod, cls.Kind(kind))
    return cls(msg_cls(spec=spec), messages_mod, kind)