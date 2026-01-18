import logging
from openstack import utils as sdk_utils
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
@staticmethod
def _find_service_by_host_and_binary(compute_client, host, binary):
    """Utility method to find a compute service by host and binary

        :param host: the name of the compute service host
        :param binary: the compute service binary, e.g. nova-compute
        :returns: novaclient.v2.services.Service dict-like object
        :raises: CommandError if no or multiple results were found
        """
    services = list(compute_client.services(host=host, binary=binary))
    if not len(services):
        msg = _('Compute service for host "%(host)s" and binary "%(binary)s" not found.') % {'host': host, 'binary': binary}
        raise exceptions.CommandError(msg)
    if len(services) > 1:
        msg = _('Multiple compute services found for host "%(host)s" and binary "%(binary)s". Unable to proceed.') % {'host': host, 'binary': binary}
        raise exceptions.CommandError(msg)
    return services[0]