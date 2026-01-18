from . import commands as _mod_commands
from . import errors, help_topics, osutils, plugin, ui, utextwrap
def _check_prefix_uniqueness(self):
    """Ensure that the index collection is able to differentiate safely."""
    prefixes = set()
    for index in self.search_path:
        prefix = index.prefix
        if prefix in prefixes:
            raise errors.DuplicateHelpPrefix(prefix)
        prefixes.add(prefix)