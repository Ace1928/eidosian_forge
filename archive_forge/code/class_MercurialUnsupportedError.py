from ... import version_info  # noqa: F401
from ... import controldir, errors
from ... import transport as _mod_transport
class MercurialUnsupportedError(errors.UnsupportedVcs):
    vcs = 'hg'
    _fmt = 'Mercurial branches are not yet supported. To interoperate with Mercurial, use the fastimport format.'