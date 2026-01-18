from ... import version_info  # noqa: F401
from ... import controldir, errors
class MonotoneUnsupportedError(errors.UnsupportedVcs):
    vcs = 'mtn'
    _fmt = 'Monotone branches are not yet supported. To interoperate with Monotone branches, use fastimport.'