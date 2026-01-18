import breezy.bzr  # noqa: F401
from breezy import controldir
from ..commands import Command, display_command
from ..option import Option, RegistryOption
class cmd_git_push_pristine_tar_deltas(Command):
    """Push pristine tar deltas to a git repository."""
    takes_options = [Option('directory', short_name='d', help='Location of repository.', type=str)]
    takes_args = ['target', 'package']

    def run(self, target, package, directory='.'):
        from ..branch import Branch
        from ..errors import CommandError, NoSuchRevision
        from ..repository import Repository
        from ..trace import warning
        from .mapping import encode_git_path
        from .object_store import get_object_store
        from .pristine_tar import revision_pristine_tar_data, store_git_pristine_tar_data
        source = Branch.open_containing(directory)[0]
        target_bzr = Repository.open(target)
        target = getattr(target_bzr, '_git', None)
        if target is None:
            raise CommandError('Target not a git repository')
        git_store = get_object_store(source.repository)
        with git_store.lock_read():
            tag_dict = source.tags.get_tag_dict()
            for name, revid in tag_dict.iteritems():
                try:
                    rev = source.repository.get_revision(revid)
                except NoSuchRevision:
                    continue
                try:
                    delta, kind = revision_pristine_tar_data(rev)
                except KeyError:
                    continue
                gitid = git_store._lookup_revision_sha1(revid)
                if not (name.startswith('upstream/') or name.startswith('upstream-')):
                    warning('Unexpected pristine tar revision tagged %s. Ignoring.', name)
                    continue
                upstream_version = name[len('upstream/'):]
                filename = '{}_{}.orig.tar.{}'.format(package, upstream_version, kind)
                if gitid not in target:
                    warning('base git id %s for %s missing in target repository', gitid, filename)
                store_git_pristine_tar_data(target, encode_git_path(filename), delta, gitid)