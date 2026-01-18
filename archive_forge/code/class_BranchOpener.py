import threading
from . import errors, trace, urlutils
from .branch import Branch
from .controldir import ControlDir, ControlDirFormat
from .transport import do_catching_redirections, get_transport
class BranchOpener:
    """Branch opener which uses a URL policy.

    All locations that are opened (stacked-on branches, references) are
    checked against a policy object.

    The policy object is expected to have the following methods:
    * check_one_url
    * should_follow_references
    * transform_fallback_location
    """
    _threading_data = threading.local()

    def __init__(self, policy, probers=None):
        """Create a new BranchOpener.

        :param policy: The opener policy to use.
        :param probers: Optional list of probers to allow.
            Defaults to local and remote bzr probers.
        """
        self.policy = policy
        self._seen_urls = set()
        if probers is None:
            probers = ControlDirFormat.all_probers()
        self.probers = probers

    @classmethod
    def install_hook(cls):
        """Install the ``transform_fallback_location`` hook.

        This is done at module import time, but transform_fallback_locationHook
        doesn't do anything unless the `_active_openers` threading.Local
        object has a 'opener' attribute in this thread.

        This is in a module-level function rather than performed at module
        level so that it can be called in setUp for testing `BranchOpener`
        as breezy.tests.TestCase.setUp clears hooks.
        """
        Branch.hooks.install_named_hook('transform_fallback_location', cls.transform_fallback_locationHook, 'BranchOpener.transform_fallback_locationHook')

    def check_and_follow_branch_reference(self, url):
        """Check URL (and possibly the referenced URL).

        This method checks that `url` passes the policy's `check_one_url`
        method, and if `url` refers to a branch reference, it checks whether
        references are allowed and whether the reference's URL passes muster
        also -- recursively, until a real branch is found.

        :param url: URL to check
        :raise BranchLoopError: If the branch references form a loop.
        :raise BranchReferenceForbidden: If this opener forbids branch
            references.
        """
        while True:
            if url in self._seen_urls:
                raise BranchLoopError()
            self._seen_urls.add(url)
            self.policy.check_one_url(url)
            next_url = self.follow_reference(url)
            if next_url is None:
                return url
            url = next_url
            if not self.policy.should_follow_references():
                raise BranchReferenceForbidden(url)

    @classmethod
    def transform_fallback_locationHook(cls, branch, url):
        """Installed as the 'transform_fallback_location' Branch hook.

        This method calls `transform_fallback_location` on the policy object
        and either returns the url it provides or passes it back to
        check_and_follow_branch_reference.
        """
        try:
            opener = getattr(cls._threading_data, 'opener')
        except AttributeError:
            return url
        new_url, check = opener.policy.transform_fallback_location(branch, url)
        if check:
            return opener.check_and_follow_branch_reference(new_url)
        else:
            return new_url

    def run_with_transform_fallback_location_hook_installed(self, callable, *args, **kw):
        if self.transform_fallback_locationHook not in Branch.hooks['transform_fallback_location']:
            raise AssertionError('hook not installed')
        self._threading_data.opener = self
        try:
            return callable(*args, **kw)
        finally:
            del self._threading_data.opener
            self._seen_urls = set()

    def _open_dir(self, url):
        """Simple BzrDir.open clone that only uses specific probers.

        :param url: URL to open
        :return: ControlDir instance
        """

        def redirected(transport, e, redirection_notice):
            self.policy.check_one_url(e.target)
            redirected_transport = transport._redirected_to(e.source, e.target)
            if redirected_transport is None:
                raise errors.NotBranchError(e.source)
            trace.note('%s is%s redirected to %s', transport.base, e.permanently, redirected_transport.base)
            return redirected_transport

        def find_format(transport):
            last_error = errors.NotBranchError(transport.base)
            for prober_kls in self.probers:
                prober = prober_kls()
                try:
                    return (transport, prober.probe_transport(transport))
                except errors.NotBranchError as e:
                    last_error = e
            else:
                raise last_error
        transport = get_transport(url)
        transport, format = do_catching_redirections(find_format, transport, redirected)
        return format.open(transport)

    def follow_reference(self, url):
        """Get the branch-reference value at the specified url.

        This exists as a separate method only to be overriden in unit tests.
        """
        controldir = self._open_dir(url)
        return controldir.get_branch_reference()

    def open(self, url, ignore_fallbacks=False):
        """Open the Bazaar branch at url, first checking it.

        What is acceptable means is defined by the policy's `follow_reference`
        and `check_one_url` methods.
        """
        if not isinstance(url, str):
            raise TypeError
        url = self.check_and_follow_branch_reference(url)

        def open_branch(url, ignore_fallbacks):
            dir = self._open_dir(url)
            return dir.open_branch(ignore_fallbacks=ignore_fallbacks)
        return self.run_with_transform_fallback_location_hook_installed(open_branch, url, ignore_fallbacks)