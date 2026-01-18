from breezy import errors, hooks
from breezy.bzr.rio import RioWriter, Stanza
from breezy.revision import NULL_REVISION
from breezy.version_info_formats import VersionInfoBuilder, create_date_str
class RioVersionInfoBuilderHooks(hooks.Hooks):
    """Hooks for rio-formatted version-info output."""

    def __init__(self):
        super().__init__('breezy.version_info_formats.format_rio', 'RioVersionInfoBuilder.hooks')
        self.add_hook('revision', 'Invoked when adding information about a revision to the RIO stanza that is printed. revision is called with a revision object and a RIO stanza.', (1, 15))