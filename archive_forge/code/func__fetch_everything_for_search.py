import operator
from .. import errors, ui
from ..i18n import gettext
from ..revision import NULL_REVISION
from ..trace import mutter
def _fetch_everything_for_search(self, search):
    """Fetch all data for the given set of revisions."""
    if self.from_repository._format.rich_root_data and (not self.to_repository._format.rich_root_data):
        raise errors.IncompatibleRepositories(self.from_repository, self.to_repository, 'different rich-root support')
    with ui.ui_factory.nested_progress_bar() as pb:
        pb.update('Get stream source')
        source = self.from_repository._get_source(self.to_repository._format)
        stream = source.get_stream(search)
        from_format = self.from_repository._format
        pb.update('Inserting stream')
        resume_tokens, missing_keys = self.sink.insert_stream(stream, from_format, [])
        if missing_keys:
            pb.update('Missing keys')
            stream = source.get_stream_for_missing_keys(missing_keys)
            pb.update('Inserting missing keys')
            resume_tokens, missing_keys = self.sink.insert_stream(stream, from_format, resume_tokens)
        if missing_keys:
            raise AssertionError('second push failed to complete a fetch {!r}.'.format(missing_keys))
        if resume_tokens:
            raise AssertionError('second push failed to commit the fetch {!r}.'.format(resume_tokens))
        pb.update('Finishing stream')
        self.sink.finished()