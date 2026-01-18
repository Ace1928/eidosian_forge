from .. import filesystem
from ..asyn import AsyncFileSystem

        Parameters
        ----------
        path: str
            Path to the directory.
        fs: AbstractFileSystem
            An instantiated filesystem to wrap.
        target_protocol, target_options:
            if fs is none, construct it from these
        fo: str
            Alternate for path; do not provide both
        