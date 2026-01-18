import base64
import contextlib
import re
from io import BytesIO
from . import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import (
from . import errors, hooks, registry
class BaseMergeDirective:
    """A request to perform a merge into a branch.

    This is the base class that all merge directive implementations
    should derive from.

    :cvar multiple_output_files: Whether or not this merge directive
        stores a set of revisions in more than one file
    """
    hooks = MergeDirectiveHooks()
    multiple_output_files = False

    def __init__(self, revision_id, testament_sha1, time, timezone, target_branch, patch=None, source_branch=None, message=None, bundle=None):
        """Constructor.

        :param revision_id: The revision to merge
        :param testament_sha1: The sha1 of the testament of the revision to
            merge.
        :param time: The current POSIX timestamp time
        :param timezone: The timezone offset
        :param target_branch: Location of branch to apply the merge to
        :param patch: The text of a diff or bundle
        :param source_branch: A public location to merge the revision from
        :param message: The message to use when committing this merge
        """
        self.revision_id = revision_id
        self.testament_sha1 = testament_sha1
        self.time = time
        self.timezone = timezone
        self.target_branch = target_branch
        self.patch = patch
        self.source_branch = source_branch
        self.message = message

    def to_lines(self):
        """Serialize as a list of lines

        :return: a list of lines
        """
        raise NotImplementedError(self.to_lines)

    def to_files(self):
        """Serialize as a set of files.

        :return: List of tuples with filename and contents as lines
        """
        raise NotImplementedError(self.to_files)

    def get_raw_bundle(self):
        """Return the bundle for this merge directive.

        :return: bundle text or None if there is no bundle
        """
        return None

    def _to_lines(self, base_revision=False):
        """Serialize as a list of lines

        :return: a list of lines
        """
        time_str = timestamp.format_patch_date(self.time, self.timezone)
        stanza = rio.Stanza(revision_id=self.revision_id, timestamp=time_str, target_branch=self.target_branch, testament_sha1=self.testament_sha1)
        for key in ('source_branch', 'message'):
            if self.__dict__[key] is not None:
                stanza.add(key, self.__dict__[key])
        if base_revision:
            stanza.add('base_revision_id', self.base_revision_id)
        lines = [b'# ' + self._format_string + b'\n']
        lines.extend(rio.to_patch_lines(stanza))
        lines.append(b'# \n')
        return lines

    def write_to_directory(self, path):
        """Write this merge directive to a series of files in a directory.

        :param path: Filesystem path to write to
        """
        raise NotImplementedError(self.write_to_directory)

    @classmethod
    def from_objects(klass, repository, revision_id, time, timezone, target_branch, patch_type='bundle', local_target_branch=None, public_branch=None, message=None):
        """Generate a merge directive from various objects

        :param repository: The repository containing the revision
        :param revision_id: The revision to merge
        :param time: The POSIX timestamp of the date the request was issued.
        :param timezone: The timezone of the request
        :param target_branch: The url of the branch to merge into
        :param patch_type: 'bundle', 'diff' or None, depending on the type of
            patch desired.
        :param local_target_branch: the submit branch, either itself or a local copy
        :param public_branch: location of a public branch containing
            the target revision.
        :param message: Message to use when committing the merge
        :return: The merge directive

        The public branch is always used if supplied.  If the patch_type is
        not 'bundle', the public branch must be supplied, and will be verified.

        If the message is not supplied, the message from revision_id will be
        used for the commit.
        """
        t_revision_id = revision_id
        if revision_id == _mod_revision.NULL_REVISION:
            t_revision_id = None
        t = testament.StrictTestament3.from_revision(repository, t_revision_id)
        if local_target_branch is None:
            submit_branch = _mod_branch.Branch.open(target_branch)
        else:
            submit_branch = local_target_branch
        if submit_branch.get_public_branch() is not None:
            target_branch = submit_branch.get_public_branch()
        if patch_type is None:
            patch = None
        else:
            submit_revision_id = submit_branch.last_revision()
            repository.fetch(submit_branch.repository, submit_revision_id)
            graph = repository.get_graph()
            ancestor_id = graph.find_unique_lca(revision_id, submit_revision_id)
            type_handler = {'bundle': klass._generate_bundle, 'diff': klass._generate_diff, None: lambda x, y, z: None}
            patch = type_handler[patch_type](repository, revision_id, ancestor_id)
        if public_branch is not None and patch_type != 'bundle':
            public_branch_obj = _mod_branch.Branch.open(public_branch)
            if not public_branch_obj.repository.has_revision(revision_id):
                raise errors.PublicBranchOutOfDate(public_branch, revision_id)
        return klass(revision_id, t.as_sha1(), time, timezone, target_branch, patch, patch_type, public_branch, message)

    def get_disk_name(self, branch):
        """Generate a suitable basename for storing this directive on disk

        :param branch: The Branch this merge directive was generated fro
        :return: A string
        """
        revno, revision_id = branch.last_revision_info()
        if self.revision_id == revision_id:
            revno = [revno]
        else:
            try:
                revno = branch.revision_id_to_dotted_revno(self.revision_id)
            except errors.NoSuchRevision:
                revno = ['merge']
        nick = re.sub('(\\W+)', '-', branch.nick).strip('-')
        return '{}-{}'.format(nick, '.'.join((str(n) for n in revno)))

    @staticmethod
    def _generate_diff(repository, revision_id, ancestor_id):
        tree_1 = repository.revision_tree(ancestor_id)
        tree_2 = repository.revision_tree(revision_id)
        s = BytesIO()
        diff.show_diff_trees(tree_1, tree_2, s, old_label='', new_label='')
        return s.getvalue()

    @staticmethod
    def _generate_bundle(repository, revision_id, ancestor_id):
        s = BytesIO()
        bundle_serializer.write_bundle(repository, revision_id, ancestor_id, s)
        return s.getvalue()

    def to_signed(self, branch):
        """Serialize as a signed string.

        :param branch: The source branch, to get the signing strategy
        :return: a string
        """
        my_gpg = gpg.GPGStrategy(branch.get_config_stack())
        return my_gpg.sign(b''.join(self.to_lines()), gpg.MODE_CLEAR)

    def to_email(self, mail_to, branch, sign=False):
        """Serialize as an email message.

        :param mail_to: The address to mail the message to
        :param branch: The source branch, to get the signing strategy and
            source email address
        :param sign: If True, gpg-sign the email
        :return: an email message
        """
        mail_from = branch.get_config_stack().get('email')
        if self.message is not None:
            subject = self.message
        else:
            revision = branch.repository.get_revision(self.revision_id)
            subject = revision.message
        if sign:
            body = self.to_signed(branch)
        else:
            body = b''.join(self.to_lines())
        message = email_message.EmailMessage(mail_from, mail_to, subject, body)
        return message

    def install_revisions(self, target_repo):
        """Install revisions and return the target revision"""
        if not target_repo.has_revision(self.revision_id):
            if self.patch_type == 'bundle':
                info = bundle_serializer.read_bundle(BytesIO(self.get_raw_bundle()))
                try:
                    info.install_revisions(target_repo, stream_input=False)
                except errors.RevisionNotPresent:
                    try:
                        submit_branch = _mod_branch.Branch.open(self.target_branch)
                    except errors.NotBranchError:
                        raise errors.TargetNotBranch(self.target_branch)
                    missing_revisions = []
                    bundle_revisions = {r.revision_id for r in info.real_revisions}
                    for revision in info.real_revisions:
                        for parent_id in revision.parent_ids:
                            if parent_id not in bundle_revisions and (not target_repo.has_revision(parent_id)):
                                missing_revisions.append(parent_id)
                    unique_missing = []
                    unique_missing_set = set()
                    for revision in reversed(missing_revisions):
                        if revision in unique_missing_set:
                            continue
                        unique_missing.append(revision)
                        unique_missing_set.add(revision)
                    for missing_revision in unique_missing:
                        target_repo.fetch(submit_branch.repository, missing_revision)
                    info.install_revisions(target_repo, stream_input=False)
            else:
                source_branch = _mod_branch.Branch.open(self.source_branch)
                target_repo.fetch(source_branch.repository, self.revision_id)
        return self.revision_id

    def compose_merge_request(self, mail_client, to, body, branch, tree=None):
        """Compose a request to merge this directive.

        :param mail_client: The mail client to use for composing this request.
        :param to: The address to compose the request to.
        :param branch: The Branch that was used to produce this directive.
        :param tree: The Tree (if any) for the Branch used to produce this
            directive.
        """
        basename = self.get_disk_name(branch)
        subject = '[MERGE] '
        if self.message is not None:
            subject += self.message
        else:
            revision = branch.repository.get_revision(self.revision_id)
            subject += revision.get_summary()
        if getattr(mail_client, 'supports_body', False):
            orig_body = body
            for hook in self.hooks['merge_request_body']:
                params = MergeRequestBodyParams(body, orig_body, self, to, basename, subject, branch, tree)
                body = hook(params)
        elif len(self.hooks['merge_request_body']) > 0:
            trace.warning('Cannot run merge_request_body hooks because mail client %s does not support message bodies.', mail_client.__class__.__name__)
        mail_client.compose_merge_request(to, subject, b''.join(self.to_lines()), basename, body)