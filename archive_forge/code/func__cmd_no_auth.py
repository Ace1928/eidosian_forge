import configparser as config_parser
import os
from tempest.lib.cli import base
def _cmd_no_auth(self, cmd, action, flags='', params=''):
    """Execute given command with noauth attributes.

        :param cmd: command to be executed
        :type cmd: string
        :param action: command on cli to run
        :type action: string
        :param flags: optional cli flags to use
        :type flags: string
        :param params: optional positional args to use
        :type params: string
        """
    flags = '--os_auth_token %(token)s --zun_url %(url)s %(flags)s' % {'token': self.os_auth_token, 'url': self.zun_url, 'flags': flags}
    return base.execute(cmd, action, flags, params, cli_dir=self.client.cli_dir)