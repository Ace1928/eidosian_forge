from .. import (branch, controldir, errors, foreign, lockable_files, lockdir,
from .. import transport as _mod_transport
from ..bzr import branch as bzrbranch
from ..bzr import bzrdir, groupcompress_repo, vf_repository
from ..bzr.pack_repo import PackCommitBuilder
class DummyForeignVcsBranch(bzrbranch.BzrBranch6, foreign.ForeignBranch):
    """A Dummy VCS Branch."""

    @property
    def user_transport(self):
        return self.controldir.user_transport

    def __init__(self, _format, _control_files, a_controldir, *args, **kwargs):
        self._format = _format
        self._base = a_controldir.transport.base
        self._ignore_fallbacks = False
        self.controldir = a_controldir
        foreign.ForeignBranch.__init__(self, DummyForeignVcsMapping(DummyForeignVcs()))
        bzrbranch.BzrBranch6.__init__(self, _format=_format, _control_files=_control_files, a_controldir=a_controldir, **kwargs)

    def _get_checkout_format(self, lightweight=False):
        """Return the most suitable metadir for a checkout of this branch.
        Weaves are used if this branch's repository uses weaves.
        """
        return self.controldir.checkout_metadir()

    def import_last_revision_info_and_tags(self, source, revno, revid, lossy=False):
        interbranch = InterToDummyVcsBranch(source, self)
        result = interbranch.push(stop_revision=revid, lossy=True)
        if lossy:
            revid = result.revidmap[revid]
        return (revno, revid)