from abc import ABCMeta
from abc import abstractmethod
class BasicLoader(TBLoader):
    """Simple TBLoader that's sufficient for most plugins."""

    def __init__(self, plugin_class):
        """Creates simple plugin instance maker.

        :param plugin_class: :class:`TBPlugin`
        """
        self.plugin_class = plugin_class

    def load(self, context):
        return self.plugin_class(context)