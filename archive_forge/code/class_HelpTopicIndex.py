import breezy
from breezy import config, i18n, osutils, registry
from another side removing lines.
class HelpTopicIndex:
    """A index for brz help that returns topics."""

    def __init__(self):
        self.prefix = ''

    def get_topics(self, topic):
        """Search for topic in the HelpTopicRegistry.

        :param topic: A topic to search for. None is treated as 'basic'.
        :return: A list which is either empty or contains a single
            RegisteredTopic entry.
        """
        if topic is None:
            topic = 'basic'
        if topic in topic_registry:
            return [RegisteredTopic(topic)]
        else:
            return []