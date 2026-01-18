from dulwich import errors as git_errors
from .. import errors as brz_errors
class GitSmartRemoteNotSupported(brz_errors.UnsupportedOperation):
    _fmt = 'This operation is not supported by the Git smart server protocol.'