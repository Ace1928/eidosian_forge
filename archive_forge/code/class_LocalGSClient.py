from ...cloudpath import CloudImplementation
from ..localclient import LocalClient
from ..localpath import LocalPath
class LocalGSClient(LocalClient):
    """Replacement for GSClient that uses the local file system. Intended as a monkeypatch
    substitute when writing tests.
    """
    _cloud_meta = local_gs_implementation