import threading
from . import errors, trace, urlutils
from .branch import Branch
from .controldir import ControlDir, ControlDirFormat
from .transport import do_catching_redirections, get_transport
class SingleSchemePolicy(BranchOpenPolicy):
    """Branch open policy that rejects URLs not on the given scheme."""

    def __init__(self, allowed_scheme):
        self.allowed_scheme = allowed_scheme

    def should_follow_references(self):
        return True

    def transform_fallback_location(self, branch, url):
        return (urlutils.join(branch.base, url), True)

    def check_one_url(self, url):
        """Check that `url` is okay to open."""
        if urlutils.URL.from_string(str(url)).scheme != self.allowed_scheme:
            raise BadUrl(url)