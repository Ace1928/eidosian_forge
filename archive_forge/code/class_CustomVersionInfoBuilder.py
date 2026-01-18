import codecs
from breezy import errors
from breezy.lazy_regex import lazy_compile
from breezy.revision import NULL_REVISION
from breezy.version_info_formats import VersionInfoBuilder, create_date_str
class CustomVersionInfoBuilder(VersionInfoBuilder):
    """Create a version file based on a custom template."""

    def generate(self, to_file):
        if self._template is None:
            raise NoTemplate()
        info = Template()
        info.add('build_date', create_date_str())
        info.add('branch_nick', self._branch.nick)
        revision_id = self._get_revision_id()
        if revision_id == NULL_REVISION:
            info.add('revno', 0)
        else:
            try:
                info.add('revno', self._get_revno_str(revision_id))
            except errors.GhostRevisionsHaveNoRevno:
                pass
            info.add('revision_id', revision_id.decode('utf-8'))
            rev = self._branch.repository.get_revision(revision_id)
            info.add('date', create_date_str(rev.timestamp, rev.timezone))
        if self._check:
            self._extract_file_revisions()
        if self._check:
            if self._clean:
                info.add('clean', 1)
            else:
                info.add('clean', 0)
        to_file.writelines(info.process(self._template))