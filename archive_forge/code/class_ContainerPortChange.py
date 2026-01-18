from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import argparse
import collections
from collections.abc import Collection, Container, Iterable, Mapping, MutableMapping
import copy
import dataclasses
import itertools
import json
import types
from typing import Any, ClassVar
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import job
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import name_generator
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.command_lib.run import secrets_mapping
from googlecloudsdk.command_lib.run import volumes
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
import six
@dataclasses.dataclass(frozen=True)
class ContainerPortChange(ContainerConfigChanger):
    """Represents the user intent to change the port name and/or number.

  Attributes:
    port: The port to set, "default" to unset the containerPort field, or None
      to not modify the port number.
    use_http2: True to set the port name for http/2, False to unset it, or None
      to not modify the port name.
    **kwargs: ContainerConfigChanger args.
  """
    port: str | None = None
    use_http2: bool | None = None

    def AdjustContainer(self, container, messages_mod):
        """Modify an existing ContainerPort or create a new one."""
        port_msg = container.ports[0] if container.ports else messages_mod.ContainerPort()
        old_port = port_msg.containerPort or 8080
        if self.port == 'default':
            port_msg.reset('containerPort')
        elif self.port is not None:
            port_msg.containerPort = int(self.port)
        if self.use_http2:
            port_msg.name = _HTTP2_NAME
        elif self.use_http2 is not None:
            port_msg.reset('name')
        if port_msg.name and (not port_msg.containerPort):
            port_msg.containerPort = _DEFAULT_PORT
        if port_msg.containerPort:
            container.ports = [port_msg]
        else:
            container.reset('ports')
        if container.startupProbe and container.startupProbe.tcpSocket:
            if container.startupProbe.tcpSocket.port == old_port:
                if port_msg.containerPort:
                    container.startupProbe.tcpSocket.port = port_msg.containerPort
                else:
                    container.startupProbe.tcpSocket.reset('port')