import fastbencode as bencode
from .. import errors, trace
from .. import transport as _mod_transport
from ..tag import Tags
class BasicTags(Tags):
    """Tag storage in an unversioned branch control file.
    """

    def set_tag(self, tag_name, tag_target):
        """Add a tag definition to the branch.

        Behaviour if the tag is already present is not defined (yet).
        """
        with self.branch.lock_write():
            master = self.branch.get_master_branch()
            if master is not None:
                master.tags.set_tag(tag_name, tag_target)
            td = self.get_tag_dict()
            td[tag_name] = tag_target
            self._set_tag_dict(td)

    def lookup_tag(self, tag_name):
        """Return the referent string of a tag"""
        td = self.get_tag_dict()
        try:
            return td[tag_name]
        except KeyError:
            raise errors.NoSuchTag(tag_name)

    def get_tag_dict(self):
        with self.branch.lock_read():
            try:
                tag_content = self.branch._get_tags_bytes()
            except _mod_transport.NoSuchFile:
                trace.warning('No branch/tags file in %s.  This branch was probably created by bzr 0.15pre.  Create an empty file to silence this message.' % (self.branch,))
                return {}
            return self._deserialize_tag_dict(tag_content)

    def delete_tag(self, tag_name):
        """Delete a tag definition.
        """
        with self.branch.lock_write():
            d = self.get_tag_dict()
            try:
                del d[tag_name]
            except KeyError:
                raise errors.NoSuchTag(tag_name)
            master = self.branch.get_master_branch()
            if master is not None:
                try:
                    master.tags.delete_tag(tag_name)
                except errors.NoSuchTag:
                    pass
            self._set_tag_dict(d)

    def _set_tag_dict(self, new_dict):
        """Replace all tag definitions

        WARNING: Calling this on an unlocked branch will lock it, and will
        replace the tags without warning on conflicts.

        :param new_dict: Dictionary from tag name to target.
        """
        return self.branch._set_tags_bytes(self._serialize_tag_dict(new_dict))

    def _serialize_tag_dict(self, tag_dict):
        td = {k.encode('utf-8'): v for k, v in tag_dict.items()}
        return bencode.bencode(td)

    def _deserialize_tag_dict(self, tag_content):
        """Convert the tag file into a dictionary of tags"""
        if tag_content == b'':
            return {}
        try:
            r = {}
            for k, v in bencode.bdecode(tag_content).items():
                r[k.decode('utf-8')] = v
            return r
        except ValueError as e:
            raise ValueError('failed to deserialize tag dictionary %r: %s' % (tag_content, e))