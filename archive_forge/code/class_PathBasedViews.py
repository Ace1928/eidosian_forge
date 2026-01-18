import re
from . import errors, osutils, transport
class PathBasedViews(_Views):
    """View storage in an unversioned tree control file.

    Views are stored in terms of paths relative to the tree root.

    The top line of the control file is a format marker in the format:

      Bazaar views format X

    where X is an integer number. After this top line, version 1 format is
    stored as follows:

     * optional name-values pairs in the format 'name=value'

     * optional view definitions, one per line in the format

       views:
       name file1 file2 ...
       name file1 file2 ...

    where the fields are separated by a nul character (\x00). The views file
    is encoded in utf-8. The only supported keyword in version 1 is
    'current' which stores the name of the current view, if any.
    """

    def __init__(self, tree):
        self.tree = tree
        self._loaded = False
        self._current = None
        self._views = {}

    def supports_views(self):
        return True

    def get_view_info(self):
        """Get the current view and dictionary of views.

        Returns: current, views where
          current = the name of the current view or None if no view is enabled
          views = a map from view name to list of files/directories
        """
        self._load_view_info()
        return (self._current, self._views)

    def set_view_info(self, current, views):
        """Set the current view and dictionary of views.

        Args:
          current: the name of the current view or None if no view is
              enabled
          views: a map from view name to list of files/directories
        """
        if current is not None and current not in views:
            raise NoSuchView(current)
        with self.tree.lock_write():
            self._current = current
            self._views = views
            self._save_view_info()

    def lookup_view(self, view_name=None):
        """Return the contents of a view.

        Args:
          view_Name: name of the view or None to lookup the current view

        Returns:
          the list of files/directories in the requested view
        """
        self._load_view_info()
        try:
            if view_name is None:
                if self._current:
                    view_name = self._current
                else:
                    return []
            return self._views[view_name]
        except KeyError:
            raise NoSuchView(view_name)

    def set_view(self, view_name, view_files, make_current=True):
        """Add or update a view definition.

        Args:
          view_name: the name of the view
          view_files: the list of files/directories in the view
          make_current: make this view the current one or not
        """
        with self.tree.lock_write():
            self._load_view_info()
            self._views[view_name] = view_files
            if make_current:
                self._current = view_name
            self._save_view_info()

    def delete_view(self, view_name):
        """Delete a view definition.

        If the view deleted is the current one, the current view is reset.
        """
        with self.tree.lock_write():
            self._load_view_info()
            try:
                del self._views[view_name]
            except KeyError:
                raise NoSuchView(view_name)
            if view_name == self._current:
                self._current = None
            self._save_view_info()

    def _save_view_info(self):
        """Save the current view and all view definitions.

        Be sure to have initialised self._current and self._views before
        calling this method.
        """
        with self.tree.lock_write():
            if self._current is None:
                keywords = {}
            else:
                keywords = {'current': self._current}
            self.tree._transport.put_bytes('views', self._serialize_view_content(keywords, self._views))

    def _load_view_info(self):
        """Load the current view and dictionary of view definitions."""
        if not self._loaded:
            with self.tree.lock_read():
                try:
                    view_content = self.tree._transport.get_bytes('views')
                except transport.NoSuchFile:
                    self._current, self._views = (None, {})
                else:
                    keywords, self._views = self._deserialize_view_content(view_content)
                    self._current = keywords.get('current')
            self._loaded = True

    def _serialize_view_content(self, keywords, view_dict):
        """Convert view keywords and a view dictionary into a stream."""
        lines = [_VIEWS_FORMAT1_MARKER]
        for key in keywords:
            line = '{}={}\n'.format(key, keywords[key])
            lines.append(line.encode('utf-8'))
        if view_dict:
            lines.append(b'views:\n')
            for view in sorted(view_dict):
                view_data = '{}\x00{}\n'.format(view, '\x00'.join(view_dict[view]))
                lines.append(view_data.encode('utf-8'))
        return b''.join(lines)

    def _deserialize_view_content(self, view_content):
        """Convert a stream into view keywords and a dictionary of views."""
        if view_content == b'':
            return ({}, {})
        lines = view_content.splitlines()
        match = _VIEWS_FORMAT_MARKER_RE.match(lines[0])
        if not match:
            raise ValueError('format marker missing from top of views file')
        elif match.group(1) != b'1':
            raise ValueError('cannot decode views format %s' % match.group(1))
        try:
            keywords = {}
            views = {}
            in_views = False
            for line in lines[1:]:
                text = line.decode('utf-8')
                if in_views:
                    parts = text.split('\x00')
                    view = parts.pop(0)
                    views[view] = parts
                elif text == 'views:':
                    in_views = True
                    continue
                elif text.find('=') >= 0:
                    keyword, value = text.split('=', 1)
                    keywords[keyword] = value
                else:
                    raise ValueError('failed to deserialize views line %s', text)
            return (keywords, views)
        except ValueError as e:
            raise ValueError('failed to deserialize views content %r: %s' % (view_content, e))