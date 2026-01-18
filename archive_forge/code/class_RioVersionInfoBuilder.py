from breezy import errors, hooks
from breezy.bzr.rio import RioWriter, Stanza
from breezy.revision import NULL_REVISION
from breezy.version_info_formats import VersionInfoBuilder, create_date_str
class RioVersionInfoBuilder(VersionInfoBuilder):
    """This writes a rio stream out."""

    def generate(self, to_file):
        info = Stanza()
        revision_id = self._get_revision_id()
        if revision_id != NULL_REVISION:
            info.add('revision-id', revision_id)
            rev = self._branch.repository.get_revision(revision_id)
            info.add('date', create_date_str(rev.timestamp, rev.timezone))
            try:
                revno = self._get_revno_str(revision_id)
            except errors.GhostRevisionsHaveNoRevno:
                revno = None
            for hook in RioVersionInfoBuilder.hooks['revision']:
                hook(rev, info)
        else:
            revno = '0'
        info.add('build-date', create_date_str())
        if revno is not None:
            info.add('revno', revno)
        if self._branch.nick is not None:
            info.add('branch-nick', self._branch.nick)
        if self._check or self._include_file_revs:
            self._extract_file_revisions()
        if self._check:
            if self._clean:
                info.add('clean', 'True')
            else:
                info.add('clean', 'False')
        if self._include_history:
            log = Stanza()
            for revision_id, message, timestamp, timezone in self._iter_revision_history():
                log.add('id', revision_id)
                log.add('message', message)
                log.add('date', create_date_str(timestamp, timezone))
            info.add('revisions', log)
        if self._include_file_revs:
            files = Stanza()
            for path in sorted(self._file_revisions.keys()):
                files.add('path', path)
                files.add('revision', self._file_revisions[path])
            info.add('file-revisions', files)
        to_file.write(info.to_string())