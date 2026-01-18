import logging
from osc_lib.command import command
from osc_lib import utils
from monascaclient import version
class MigratingCommand(command.Command, metaclass=MigratingCommandMeta):
    """MigratingCommand is temporary command.

    MigratingCommand allows to map function defined
    shell commands from :py:module:`monascaclient.v2_0.shell`
    into :py:class:`command.Command` instances.

    Note:
        This class is temporary solution during migrating
        to osc_lib and will be removed when all
        shell commands are migrated to cliff commands.

    """
    _help = None
    _args = None
    _callback = None

    def __init__(self, app, app_args, cmd_name=None):
        super(MigratingCommand, self).__init__(app, app_args, cmd_name)
        self._client = None
        self._endpoint = None

    def take_action(self, parsed_args):
        return self._callback(self.mon_client, parsed_args)

    def get_parser(self, prog_name):
        parser = super(MigratingCommand, self).get_parser(prog_name)
        for args, kwargs in self._args:
            parser.add_argument(*args, **kwargs)
        parser.add_argument('-j', '--json', action='store_true', help='output raw json response')
        return parser

    @property
    def mon_client(self):
        if not self._client:
            self.log.debug('Initializing mon-client')
            self._client = make_client(api_version=self.mon_version, endpoint=self.mon_url, session=self.app.client_manager.session)
        return self._client

    @property
    def mon_version(self):
        return self.app_args.monasca_api_version

    @property
    def mon_url(self):
        if self._endpoint:
            return self._endpoint
        app_args = self.app_args
        cm = self.app.client_manager
        endpoint = app_args.monasca_api_url
        if not endpoint:
            req_data = {'service_type': 'monitoring', 'region_name': cm.region_name, 'interface': cm.interface}
            LOG.debug('Discovering monasca endpoint using %s' % req_data)
            endpoint = cm.get_endpoint_for_service_type(**req_data)
        else:
            LOG.debug('Using supplied endpoint=%s' % endpoint)
        self._endpoint = endpoint
        return self._endpoint