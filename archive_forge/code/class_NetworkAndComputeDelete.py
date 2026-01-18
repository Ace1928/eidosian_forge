import abc
import contextlib
import logging
import openstack.exceptions
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from openstackclient.i18n import _
from openstackclient.network import utils
class NetworkAndComputeDelete(NetworkAndComputeCommand, metaclass=abc.ABCMeta):
    """Network and Compute Delete

    Delete class for commands that support implementation via
    the network or compute endpoint. Such commands have different
    implementations for take_action() and may even have different
    arguments. This class supports bulk deletion, and error handling
    following the rules in doc/source/command-errors.rst.
    """

    def take_action(self, parsed_args):
        ret = 0
        resources = getattr(parsed_args, self.resource, [])
        for r in resources:
            self.r = r
            try:
                if self.app.client_manager.is_network_endpoint_enabled():
                    self.take_action_network(self.app.client_manager.network, parsed_args)
                else:
                    self.take_action_compute(self.app.client_manager.compute, parsed_args)
            except Exception as e:
                msg = _("Failed to delete %(resource)s with name or ID '%(name_or_id)s': %(e)s") % {'resource': self.resource, 'name_or_id': r, 'e': e}
                LOG.error(msg)
                ret += 1
        if ret:
            total = len(resources)
            msg = _('%(num)s of %(total)s %(resource)ss failed to delete.') % {'num': ret, 'total': total, 'resource': self.resource}
            raise exceptions.CommandError(msg)