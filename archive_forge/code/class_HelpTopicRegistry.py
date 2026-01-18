import breezy
from breezy import config, i18n, osutils, registry
from another side removing lines.
class HelpTopicRegistry(registry.Registry):
    """A Registry customized for handling help topics."""

    def register(self, topic, detail, summary, section=SECT_LIST):
        """Register a new help topic.

        :param topic: Name of documentation entry
        :param detail: Function or string object providing detailed
            documentation for topic.  Function interface is detail(topic).
            This should return a text string of the detailed information.
            See the module documentation for details on help text formatting.
        :param summary: String providing single-line documentation for topic.
        :param section: Section in reference manual - see SECT_* identifiers.
        """
        info = (summary, section)
        super().register(topic, detail, info=info)

    def register_lazy(self, topic, module_name, member_name, summary, section=SECT_LIST):
        """Register a new help topic, and import the details on demand.

        :param topic: Name of documentation entry
        :param module_name: The module to find the detailed help.
        :param member_name: The member of the module to use for detailed help.
        :param summary: String providing single-line documentation for topic.
        :param section: Section in reference manual - see SECT_* identifiers.
        """
        info = (summary, section)
        super().register_lazy(topic, module_name, member_name, info=info)

    def get_detail(self, topic):
        """Get the detailed help on a given topic."""
        obj = self.get(topic)
        if callable(obj):
            return obj(topic)
        else:
            return obj

    def get_summary(self, topic):
        """Get the single line summary for the topic."""
        info = self.get_info(topic)
        if info is None:
            return None
        else:
            return info[0]

    def get_section(self, topic):
        """Get the section for the topic."""
        info = self.get_info(topic)
        if info is None:
            return None
        else:
            return info[1]

    def get_topics_for_section(self, section):
        """Get the set of topics in a section."""
        result = set()
        for topic in self.keys():
            if section == self.get_section(topic):
                result.add(topic)
        return result