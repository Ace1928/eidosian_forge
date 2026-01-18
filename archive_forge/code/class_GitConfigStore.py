from .. import config
class GitConfigStore(config.Store):
    """Store that uses gitconfig."""

    def __init__(self, id, config):
        self.id = id
        self._config = config

    def get_sections(self):
        return [(self, GitConfigSectionDefault('default', self._config))]