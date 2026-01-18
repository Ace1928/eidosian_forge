from . import commands as _mod_commands
from . import errors, help_topics, osutils, plugin, ui, utextwrap
class NoHelpTopic(errors.BzrError):
    _fmt = "No help could be found for '%(topic)s'. Please use 'brz help topics' to obtain a list of topics."

    def __init__(self, topic):
        self.topic = topic