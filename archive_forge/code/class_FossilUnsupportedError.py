from ... import version_info  # noqa: F401
from ... import controldir, errors
class FossilUnsupportedError(errors.UnsupportedVcs):
    vcs = 'fossil'
    _fmt = 'Fossil branches are not yet supported. To interoperate with Fossil branches, use fastimport.'