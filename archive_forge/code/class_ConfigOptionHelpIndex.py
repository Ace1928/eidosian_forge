import breezy
from breezy import config, i18n, osutils, registry
from another side removing lines.
class ConfigOptionHelpIndex:
    """A help index that returns help topics for config options."""

    def __init__(self):
        self.prefix = 'configuration/'

    def get_topics(self, topic):
        """Search for topic in the registered config options.

        :param topic: A topic to search for.
        :return: A list which is either empty or contains a single
            config.Option entry.
        """
        if topic is None:
            return []
        elif topic.startswith(self.prefix):
            topic = topic[len(self.prefix):]
        if topic in config.option_registry:
            return [config.option_registry.get(topic)]
        else:
            return []