import threading
from . import errors, trace, urlutils
from .branch import Branch
from .controldir import ControlDir, ControlDirFormat
from .transport import do_catching_redirections, get_transport
class WhitelistPolicy(BranchOpenPolicy):
    """Branch policy that only allows certain URLs."""

    def __init__(self, should_follow_references, allowed_urls=None, check=False):
        if allowed_urls is None:
            allowed_urls = []
        self.allowed_urls = {url.rstrip('/') for url in allowed_urls}
        self.check = check

    def should_follow_references(self):
        return self._should_follow_references

    def check_one_url(self, url):
        if url.rstrip('/') not in self.allowed_urls:
            raise BadUrl(url)

    def transform_fallback_location(self, branch, url):
        """See `BranchOpenPolicy.transform_fallback_location`.

        Here we return the URL that would be used anyway and optionally check
        it.
        """
        return (urlutils.join(branch.base, url), self.check)