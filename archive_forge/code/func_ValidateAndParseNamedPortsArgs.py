from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
import six
from six.moves import range  # pylint: disable=redefined-builtin
def ValidateAndParseNamedPortsArgs(messages, named_ports):
    """Validates named ports flags."""
    ports = []
    for named_port in named_ports:
        if named_port.count(':') != 1:
            raise calliope_exceptions.InvalidArgumentException(named_port, 'Named ports should follow NAME:PORT format.')
        host, port = named_port.split(':')
        if not port.isdigit():
            raise calliope_exceptions.InvalidArgumentException(named_port, 'Named ports should follow NAME:PORT format.')
        ports.append(messages.NamedPort(name=host, port=int(port)))
    return ports