import re
import patiencediff
from ... import bugtracker, osutils
class CommitTemplate:

    def __init__(self, commit, message, filespec):
        """Create a commit template for commit with initial message message.

        :param commit: A Commit object for the in progress commit.
        :param message: The current message (which may be None).
        :param filespec: List of files to match
        """
        self.commit = commit
        self.message = message
        self.filespec = filespec

    def make(self):
        """Make the template.

        If NEWS is missing or not not modified, the original template is
        returned unaltered. Otherwise the changes from NEWS are concatenated
        with whatever message was provided to __init__.
        """
        delta = self.commit.builder.get_basis_delta()
        found_old_path = None
        found_entry = None
        for old_path, new_path, fileid, entry in delta:
            if new_path in self.filespec:
                found_entry = entry
                found_old_path = old_path
                break
        if not found_entry:
            return self.message
        if found_old_path is None:
            _, new_chunks = next(self.commit.builder.repository.iter_files_bytes([(found_entry.file_id, found_entry.revision, None)]))
            content = b''.join(new_chunks).decode('utf-8')
            return self.merge_message(content)
        else:
            old_revision = self.commit.basis_tree.get_file_revision(old_path)
            needed = [(found_entry.file_id, found_entry.revision, 'new'), (found_entry.file_id, old_revision, 'old')]
            contents = self.commit.builder.repository.iter_files_bytes(needed)
            lines = {}
            for name, chunks in contents:
                lines[name] = osutils.chunks_to_lines(list(chunks))
            new = lines['new']
            sequence_matcher = patiencediff.PatienceSequenceMatcher(None, lines['old'], new)
            new_lines = []
            for group in sequence_matcher.get_opcodes():
                tag, i1, i2, j1, j2 = group
                if tag == 'equal':
                    continue
                if tag == 'delete':
                    continue
                new_lines.extend([l.decode('utf-8') for l in new[j1:j2]])
            if not self.commit.revprops.get('bugs'):
                bt = bugtracker.tracker_registry.get('launchpad')
                bugids = []
                for line in new_lines:
                    bugids.extend(_BUG_MATCH.findall(line))
                self.commit.revprops['bugs'] = bugtracker.encode_fixes_bug_urls([(bt.get_bug_url(bugid), bugtracker.FIXED) for bugid in bugids])
            return self.merge_message(''.join(new_lines))

    def merge_message(self, new_message):
        """Merge new_message with self.message.

        :param new_message: A string message to merge with self.message.
        :return: A string with the merged messages.
        """
        if self.message is None:
            return new_message
        return self.message + new_message