import threading
from . import errors, trace, urlutils
from .branch import Branch
from .controldir import ControlDir, ControlDirFormat
from .transport import do_catching_redirections, get_transport
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