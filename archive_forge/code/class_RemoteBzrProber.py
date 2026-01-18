from typing import TYPE_CHECKING
from .. import config, controldir, errors, pyutils, registry
from .. import transport as _mod_transport
from ..branch import format_registry as branch_format_registry
from ..repository import format_registry as repository_format_registry
from ..workingtree import format_registry as workingtree_format_registry
class RemoteBzrProber(controldir.Prober):
    """Prober for remote servers that provide a Bazaar smart server."""

    @classmethod
    def priority(klass, transport):
        return -10

    @classmethod
    def probe_transport(klass, transport):
        """Return a RemoteBzrDirFormat object if it looks possible."""
        try:
            medium = transport.get_smart_medium()
        except (NotImplementedError, AttributeError, errors.TransportNotPossible, errors.NoSmartMedium, errors.SmartProtocolError):
            raise errors.NotBranchError(path=transport.base)
        else:
            if medium.should_probe():
                try:
                    server_version = medium.protocol_version()
                except errors.SmartProtocolError:
                    raise errors.NotBranchError(path=transport.base)
                if server_version != '2':
                    raise errors.NotBranchError(path=transport.base)
            from .remote import RemoteBzrDirFormat
            return RemoteBzrDirFormat()

    @classmethod
    def known_formats(cls):
        from .remote import RemoteBzrDirFormat
        return [RemoteBzrDirFormat()]