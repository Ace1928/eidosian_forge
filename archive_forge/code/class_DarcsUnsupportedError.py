from breezy import controldir, errors
from ... import version_info  # noqa: F401
class DarcsUnsupportedError(errors.UnsupportedVcs):
    vcs = 'darcs'
    _fmt = 'Darcs branches are not yet supported. To interoperate with darcs branches, use fastimport.'