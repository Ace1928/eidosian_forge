from .object_store import iter_tree_contents
from .objects import Blob
from .patch import is_binary
class BlobNormalizer:
    """An object to store computation result of which filter to apply based
    on configuration, gitattributes, path and operation (checkin or checkout).
    """

    def __init__(self, config_stack, gitattributes) -> None:
        self.config_stack = config_stack
        self.gitattributes = gitattributes
        try:
            core_eol = config_stack.get('core', 'eol')
        except KeyError:
            core_eol = 'native'
        try:
            core_autocrlf = config_stack.get('core', 'autocrlf').lower()
        except KeyError:
            core_autocrlf = False
        self.fallback_read_filter = get_checkout_filter(core_eol, core_autocrlf, self.gitattributes)
        self.fallback_write_filter = get_checkin_filter(core_eol, core_autocrlf, self.gitattributes)

    def checkin_normalize(self, blob, tree_path):
        """Normalize a blob during a checkin operation."""
        if self.fallback_write_filter is not None:
            return normalize_blob(blob, self.fallback_write_filter, binary_detection=True)
        return blob

    def checkout_normalize(self, blob, tree_path):
        """Normalize a blob during a checkout operation."""
        if self.fallback_read_filter is not None:
            return normalize_blob(blob, self.fallback_read_filter, binary_detection=True)
        return blob