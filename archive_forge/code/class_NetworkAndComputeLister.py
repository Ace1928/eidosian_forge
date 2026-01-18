import abc
import contextlib
import logging
import openstack.exceptions
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from openstackclient.i18n import _
from openstackclient.network import utils
class NetworkAndComputeLister(NetDetectionMixin, command.Lister, metaclass=abc.ABCMeta):
    """Network and Compute Lister

    Lister class for commands that support implementation via
    the network or compute endpoint. Such commands have different
    implementations for take_action() and may even have different
    arguments.
    """
    pass