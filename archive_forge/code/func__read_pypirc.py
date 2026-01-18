import os
from configparser import RawConfigParser
import warnings
from distutils.cmd import Command
def _read_pypirc(self):
    """Reads the .pypirc file."""
    rc = self._get_rc_file()
    if os.path.exists(rc):
        self.announce('Using PyPI login from %s' % rc)
        repository = self.repository or self.DEFAULT_REPOSITORY
        config = RawConfigParser()
        config.read(rc)
        sections = config.sections()
        if 'distutils' in sections:
            index_servers = config.get('distutils', 'index-servers')
            _servers = [server.strip() for server in index_servers.split('\n') if server.strip() != '']
            if _servers == []:
                if 'pypi' in sections:
                    _servers = ['pypi']
                else:
                    return {}
            for server in _servers:
                current = {'server': server}
                current['username'] = config.get(server, 'username')
                for key, default in (('repository', self.DEFAULT_REPOSITORY), ('realm', self.DEFAULT_REALM), ('password', None)):
                    if config.has_option(server, key):
                        current[key] = config.get(server, key)
                    else:
                        current[key] = default
                if server == 'pypi' and repository in (self.DEFAULT_REPOSITORY, 'pypi'):
                    current['repository'] = self.DEFAULT_REPOSITORY
                    return current
                if current['server'] == repository or current['repository'] == repository:
                    return current
        elif 'server-login' in sections:
            server = 'server-login'
            if config.has_option(server, 'repository'):
                repository = config.get(server, 'repository')
            else:
                repository = self.DEFAULT_REPOSITORY
            return {'username': config.get(server, 'username'), 'password': config.get(server, 'password'), 'repository': repository, 'server': server, 'realm': self.DEFAULT_REALM}
    return {}