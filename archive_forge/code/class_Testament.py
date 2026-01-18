from copy import copy
from ..osutils import contains_linebreaks, contains_whitespace, sha_strings
from ..tree import Tree
class Testament:
    """Reduced summary of a revision.

    Testaments can be

      - produced from a revision
      - written to a stream
      - loaded from a stream
      - compared to a revision
    """
    long_header = 'bazaar-ng testament version 1\n'
    short_header = 'bazaar-ng testament short form 1\n'
    include_root = False

    @classmethod
    def from_revision(cls, repository, revision_id):
        """Produce a new testament from a historical revision."""
        rev = repository.get_revision(revision_id)
        tree = repository.revision_tree(revision_id)
        return cls(rev, tree)

    @classmethod
    def from_revision_tree(cls, tree):
        """Produce a new testament from a revision tree."""
        rev = tree._repository.get_revision(tree.get_revision_id())
        return cls(rev, tree)

    def __init__(self, rev, tree):
        """Create a new testament for rev using tree."""
        self.revision_id = rev.revision_id
        self.committer = rev.committer
        self.timezone = rev.timezone or 0
        self.timestamp = rev.timestamp
        self.message = rev.message
        self.parent_ids = rev.parent_ids[:]
        if not isinstance(tree, Tree):
            raise TypeError('As of bzr 2.4 Testament.__init__() takes a Revision and a Tree.')
        self.tree = tree
        self.revprops = copy(rev.properties)
        if contains_whitespace(self.revision_id):
            raise ValueError(self.revision_id)
        if contains_linebreaks(self.committer):
            raise ValueError(self.committer)

    def as_text_lines(self):
        """Yield text form as a sequence of lines.

        The result is returned in utf-8, because it should be signed or
        hashed in that encoding.
        """
        r = []
        a = r.append
        a(self.long_header)
        a('revision-id: %s\n' % self.revision_id.decode('utf-8'))
        a('committer: %s\n' % self.committer)
        a('timestamp: %d\n' % self.timestamp)
        a('timezone: %d\n' % self.timezone)
        a('parents:\n')
        for parent_id in sorted(self.parent_ids):
            if contains_whitespace(parent_id):
                raise ValueError(parent_id)
            a('  %s\n' % parent_id.decode('utf-8'))
        a('message:\n')
        for l in self.message.splitlines():
            a('  %s\n' % l)
        a('inventory:\n')
        for path, ie in self._get_entries():
            a(self._entry_to_line(path, ie))
        r.extend(self._revprops_to_lines())
        return [line.encode('utf-8') for line in r]

    def _get_entries(self):
        return ((path, ie) for path, file_class, kind, ie in self.tree.list_files(include_root=self.include_root))

    def _escape_path(self, path):
        if contains_linebreaks(path):
            raise ValueError(path)
        if not isinstance(path, str):
            path = path.decode('ascii')
        return path.replace('\\', '/').replace(' ', '\\ ')

    def _entry_to_line(self, path, ie):
        """Turn an inventory entry into a testament line"""
        if contains_whitespace(ie.file_id):
            raise ValueError(ie.file_id)
        content = ''
        content_spacer = ''
        if ie.kind == 'file':
            if not ie.text_sha1:
                raise AssertionError()
            content = ie.text_sha1.decode('ascii')
            content_spacer = ' '
        elif ie.kind == 'symlink':
            if not ie.symlink_target:
                raise AssertionError()
            content = self._escape_path(ie.symlink_target)
            content_spacer = ' '
        l = '  {} {} {}{}{}\n'.format(ie.kind, self._escape_path(path), ie.file_id.decode('utf8'), content_spacer, content)
        return l

    def as_text(self):
        return b''.join(self.as_text_lines())

    def as_short_text(self):
        """Return short digest-based testament."""
        return self.short_header.encode('ascii') + b'revision-id: %s\nsha1: %s\n' % (self.revision_id, self.as_sha1())

    def _revprops_to_lines(self):
        """Pack up revision properties."""
        if not self.revprops:
            return []
        r = ['properties:\n']
        for name, value in sorted(self.revprops.items()):
            if contains_whitespace(name):
                raise ValueError(name)
            r.append('  %s:\n' % name)
            for line in value.splitlines():
                r.append('    %s\n' % line)
        return r

    def as_sha1(self):
        return sha_strings(self.as_text_lines())