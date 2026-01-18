from cliff import lister
from vitrageclient.common import utils
class ServiceList(lister.Lister):
    """List all services"""

    def get_parser(self, prog_name):
        parser = super(ServiceList, self).get_parser(prog_name)
        return parser

    def take_action(self, parsed_args):
        service = utils.get_client(self).service.list()
        return utils.list2cols_with_rename((('Name', 'name'), ('Process Id', 'process'), ('Hostname', 'hostname'), ('Created At', 'created')), service)