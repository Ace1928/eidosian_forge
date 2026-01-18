import os.path
class IconCache(object):
    """Maintain a cache of icons.  If an icon is used more than once by a GUI
    then ensure that only one copy is created.
    """

    def __init__(self, object_factory, qtgui_module):
        """Initialise the cache."""
        self._object_factory = object_factory
        self._qtgui_module = qtgui_module
        self._base_dir = ''
        self._cache = []

    def set_base_dir(self, base_dir):
        """ Set the base directory to be used for all relative filenames. """
        self._base_dir = base_dir

    def get_icon(self, iconset):
        """Return an icon described by the given iconset tag."""
        theme = iconset.attrib.get('theme')
        if theme is not None:
            return self._object_factory.createQObject('QIcon.fromTheme', 'icon', (self._object_factory.asString(theme),), is_attribute=False)
        if iconset.text is None:
            return None
        iset = _IconSet(iconset, self._base_dir)
        try:
            idx = self._cache.index(iset)
        except ValueError:
            idx = -1
        if idx >= 0:
            iset = self._cache[idx]
        else:
            name = 'icon'
            idx = len(self._cache)
            if idx > 0:
                name += str(idx)
            icon = self._object_factory.createQObject('QIcon', name, (), is_attribute=False)
            iset.set_icon(icon, self._qtgui_module)
            self._cache.append(iset)
        return iset.icon