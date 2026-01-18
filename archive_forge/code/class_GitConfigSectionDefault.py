from .. import config
class GitConfigSectionDefault(config.Section):
    """The "default" config section in git config file"""
    id = None

    def __init__(self, id, config):
        self._config = config

    def get(self, name, default=None, expand=True):
        if name == 'email':
            try:
                email = self._config.get((b'user',), b'email')
            except KeyError:
                return None
            try:
                name = self._config.get((b'user',), b'name')
            except KeyError:
                return email.decode()
            return '{} <{}>'.format(name.decode(), email.decode())
        if name == 'gpg_signing_key':
            try:
                key = self._config.get((b'user',), b'signingkey')
            except KeyError:
                return None
            return key.decode()
        return None

    def iter_option_names(self):
        try:
            self._config.get((b'user',), b'email')
        except KeyError:
            pass
        else:
            yield 'email'
        try:
            self._config.get((b'user',), b'signingkey')
        except KeyError:
            pass
        else:
            yield 'gpg_signing_key'