from stat import S_ISDIR
from .. import errors
from .. import transport as _mod_transport
from .. import urlutils
from . import decorator
See Transport.rename().

        This variation on rename converts DirectoryNotEmpty and FileExists
        errors into ResourceBusy if the target is a directory.
        