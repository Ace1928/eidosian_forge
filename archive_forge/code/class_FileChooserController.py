from weakref import ref
from time import time
from kivy.core.text import DEFAULT_FONT
from kivy.compat import string_types
from kivy.factory import Factory
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.utils import platform as core_platform
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import (
import collections.abc
from os import listdir
from os.path import (
from fnmatch import fnmatch
class FileChooserController(RelativeLayout):
    """Base for implementing a FileChooser. Don't use this class directly, but
    prefer using an implementation such as the :class:`FileChooser`,
    :class:`FileChooserListView` or :class:`FileChooserIconView`.

    :Events:
        `on_entry_added`: entry, parent
            Fired when a root-level entry is added to the file list. If you
            return True from this event, the entry is not added to FileChooser.
        `on_entries_cleared`
            Fired when the the entries list is cleared, usually when the
            root is refreshed.
        `on_subentry_to_entry`: entry, parent
            Fired when a sub-entry is added to an existing entry or
            when entries are removed from an entry e.g. when
            a node is closed.
        `on_submit`: selection, touch
            Fired when a file has been selected with a double-tap.
    """
    _ENTRY_TEMPLATE = None
    layout = ObjectProperty(baseclass=FileChooserLayout)
    '\n    Reference to the layout widget instance.\n\n    layout is an :class:`~kivy.properties.ObjectProperty`.\n\n    .. versionadded:: 1.9.0\n    '
    path = StringProperty(u'/')
    '\n    path is a :class:`~kivy.properties.StringProperty` and defaults to the\n    current working directory as a unicode string. It specifies the path on the\n    filesystem that this controller should refer to.\n\n    .. warning::\n\n        If a unicode path is specified, all the files returned will be in\n        unicode, allowing the display of unicode files and paths. If a bytes\n        path is specified, only files and paths with ascii names will be\n        displayed properly: non-ascii filenames will be displayed and listed\n        with questions marks (?) instead of their unicode characters.\n    '
    filters = ListProperty([])
    "\n    filters specifies the filters to be applied to the files in the directory.\n    filters is a :class:`~kivy.properties.ListProperty` and defaults to [].\n    This is equivalent to '\\*' i.e. nothing is filtered.\n\n    The filters are not reset when the path changes. You need to do that\n    yourself if desired.\n\n    There are two kinds of filters: patterns and callbacks.\n\n    #. Patterns\n\n        e.g. ['\\*.png'].\n        You can use the following patterns:\n\n            ========== =================================\n            Pattern     Meaning\n            ========== =================================\n            \\*         matches everything\n            ?          matches any single character\n            [seq]      matches any character in seq\n            [!seq]     matches any character not in seq\n            ========== =================================\n\n    #. Callbacks\n\n        You can specify a function that will be called for each file. The\n        callback will be passed the folder and file name as the first\n        and second parameters respectively. It should return True to\n        indicate a match and False otherwise.\n\n    .. versionchanged:: 1.4.0\n        Added the option to specify the filter as a callback.\n    "
    filter_dirs = BooleanProperty(False)
    '\n    Indicates whether filters should also apply to directories.\n    filter_dirs is a :class:`~kivy.properties.BooleanProperty` and defaults to\n    False.\n    '
    sort_func = ObjectProperty(alphanumeric_folders_first)
    '\n    Provides a function to be called with a list of filenames as the first\n    argument and the filesystem implementation as the second argument. It\n    returns a list of filenames sorted for display in the view.\n\n    sort_func is an :class:`~kivy.properties.ObjectProperty` and defaults to a\n    function returning alphanumerically named folders first.\n\n    .. versionchanged:: 1.8.0\n\n        The signature needs now 2 arguments: first the list of files,\n        second the filesystem class to use.\n    '
    files = ListProperty([])
    '\n    The list of files in the directory specified by path after applying the\n    filters.\n\n    files is a read-only :class:`~kivy.properties.ListProperty`.\n    '
    show_hidden = BooleanProperty(False)
    '\n    Determines whether hidden files and folders should be shown.\n\n    show_hidden is a :class:`~kivy.properties.BooleanProperty` and defaults to\n    False.\n    '
    selection = ListProperty([])
    '\n    Contains the list of files that are currently selected.\n\n    selection is a read-only :class:`~kivy.properties.ListProperty` and\n    defaults to [].\n    '
    multiselect = BooleanProperty(False)
    '\n    Determines whether the user is able to select multiple files or not.\n\n    multiselect is a :class:`~kivy.properties.BooleanProperty` and defaults to\n    False.\n    '
    dirselect = BooleanProperty(False)
    '\n    Determines whether directories are valid selections or not.\n\n    dirselect is a :class:`~kivy.properties.BooleanProperty` and defaults to\n    False.\n\n    .. versionadded:: 1.1.0\n    '
    rootpath = StringProperty(None, allownone=True)
    '\n    Root path to use instead of the system root path. If set, it will not show\n    a ".." directory to go up to the root path. For example, if you set\n    rootpath to /users/foo, the user will be unable to go to /users or to any\n    other directory not starting with /users/foo.\n\n    rootpath is a :class:`~kivy.properties.StringProperty` and defaults\n    to None.\n\n    .. versionadded:: 1.2.0\n\n    .. note::\n\n        Similarly to :attr:`path`, whether `rootpath` is specified as\n        bytes or a unicode string determines the type of the filenames and\n        paths read.\n    '
    progress_cls = ObjectProperty(FileChooserProgress)
    'Class to use for displaying a progress indicator for filechooser\n    loading.\n\n    progress_cls is an :class:`~kivy.properties.ObjectProperty` and defaults to\n    :class:`FileChooserProgress`.\n\n    .. versionadded:: 1.2.0\n\n    .. versionchanged:: 1.8.0\n\n        If set to a string, the :class:`~kivy.factory.Factory` will be used to\n        resolve the class name.\n\n    '
    file_encodings = ListProperty(['utf-8', 'latin1', 'cp1252'], deprecated=True)
    "Possible encodings for decoding a filename to unicode. In the case that\n    the user has a non-ascii filename, undecodable without knowing its\n    initial encoding, we have no other choice than to guess it.\n\n    Please note that if you encounter an issue because of a missing encoding\n    here, we'll be glad to add it to this list.\n\n    file_encodings is a :class:`~kivy.properties.ListProperty` and defaults to\n    ['utf-8', 'latin1', 'cp1252'].\n\n    .. versionadded:: 1.3.0\n\n    .. deprecated:: 1.8.0\n       This property is no longer used as the filechooser no longer decodes\n       the file names.\n\n    "
    file_system = ObjectProperty(FileSystemLocal(), baseclass=FileSystemAbstract)
    'The file system object used to access the file system. This should be a\n    subclass of :class:`FileSystemAbstract`.\n\n    file_system is an :class:`~kivy.properties.ObjectProperty` and defaults to\n    :class:`FileSystemLocal()`\n\n    .. versionadded:: 1.8.0\n    '
    font_name = StringProperty(DEFAULT_FONT)
    "Filename of the font to use in UI components. The path can be\n    absolute or relative.  Relative paths are resolved by the\n    :func:`~kivy.resources.resource_find` function.\n\n    :attr:`font_name` is a :class:`~kivy.properties.StringProperty` and\n    defaults to 'Roboto'. This value is taken\n    from :class:`~kivy.config.Config`.\n    "
    _update_files_ev = None
    _create_files_entries_ev = None
    __events__ = ('on_entry_added', 'on_entries_cleared', 'on_subentry_to_entry', 'on_remove_subentry', 'on_submit')

    def __init__(self, **kwargs):
        self._progress = None
        super(FileChooserController, self).__init__(**kwargs)
        self._items = []
        fbind = self.fbind
        fbind('selection', self._update_item_selection)
        self._previous_path = [self.path]
        fbind('path', self._save_previous_path)
        update = self._trigger_update
        fbind('path', update)
        fbind('filters', update)
        fbind('rootpath', update)
        update()

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return
        if self.disabled:
            return True
        return super(FileChooserController, self).on_touch_down(touch)

    def on_touch_up(self, touch):
        if not self.collide_point(*touch.pos):
            return
        if self.disabled:
            return True
        return super(FileChooserController, self).on_touch_up(touch)

    def _update_item_selection(self, *args):
        for item in self._items:
            item.selected = item.path in self.selection

    def _save_previous_path(self, instance, value):
        self._previous_path.append(value)
        self._previous_path = self._previous_path[-2:]

    def _trigger_update(self, *args):
        ev = self._update_files_ev
        if ev is None:
            ev = self._update_files_ev = Clock.create_trigger(self._update_files)
        ev()

    def on_entry_added(self, node, parent=None):
        if self.layout:
            self.layout.dispatch('on_entry_added', node, parent)

    def on_entries_cleared(self):
        if self.layout:
            self.layout.dispatch('on_entries_cleared')

    def on_subentry_to_entry(self, subentry, entry):
        if self.layout:
            self.layout.dispatch('on_subentry_to_entry', subentry, entry)

    def on_remove_subentry(self, subentry, entry):
        if self.layout:
            self.layout.dispatch('on_remove_subentry', subentry, entry)

    def on_submit(self, selected, touch=None):
        if self.layout:
            self.layout.dispatch('on_submit', selected, touch)

    def entry_touched(self, entry, touch):
        """(internal) This method must be called by the template when an entry
        is touched by the user.
        """
        if 'button' in touch.profile and touch.button in ('scrollup', 'scrolldown', 'scrollleft', 'scrollright'):
            return False
        _dir = self.file_system.is_dir(entry.path)
        dirselect = self.dirselect
        if _dir and dirselect and touch.is_double_tap:
            self.open_entry(entry)
            return
        if self.multiselect:
            if entry.path in self.selection:
                self.selection.remove(entry.path)
            else:
                if _dir and (not self.dirselect):
                    self.open_entry(entry)
                    return
                self.selection.append(entry.path)
        else:
            if _dir and (not self.dirselect):
                return
            self.selection = [abspath(join(self.path, entry.path))]

    def entry_released(self, entry, touch):
        """(internal) This method must be called by the template when an entry
        is touched by the user.

        .. versionadded:: 1.1.0
        """
        if 'button' in touch.profile and touch.button in ('scrollup', 'scrolldown', 'scrollleft', 'scrollright'):
            return False
        if not self.multiselect:
            if self.file_system.is_dir(entry.path) and (not self.dirselect):
                self.open_entry(entry)
            elif touch.is_double_tap:
                if self.dirselect and self.file_system.is_dir(entry.path):
                    return
                else:
                    self.dispatch('on_submit', self.selection, touch)

    def open_entry(self, entry):
        try:
            self.file_system.listdir(entry.path)
        except OSError:
            entry.locked = True
        else:
            self.path = abspath(join(self.path, entry.path))
            self.selection = [self.path] if self.dirselect else []

    def _apply_filters(self, files):
        if not self.filters:
            return files
        filtered = []
        for filt in self.filters:
            if isinstance(filt, collections.abc.Callable):
                filtered.extend([fn for fn in files if filt(self.path, fn)])
            else:
                filtered.extend([fn for fn in files if fnmatch(fn, filt)])
        if not self.filter_dirs:
            dirs = [fn for fn in files if self.file_system.is_dir(fn)]
            filtered.extend(dirs)
        return list(set(filtered))

    def get_nice_size(self, fn):
        """Pass the filepath. Returns the size in the best human readable
        format or '' if it is a directory (Don't recursively calculate size).
        """
        if self.file_system.is_dir(fn):
            return ''
        try:
            size = self.file_system.getsize(fn)
        except OSError:
            return '--'
        for unit in filesize_units:
            if size < 1024.0:
                return '%1.0f %s' % (size, unit)
            size /= 1024.0

    def _update_files(self, *args, **kwargs):
        self._gitems = []
        self._gitems_parent = kwargs.get('parent', None)
        self._gitems_gen = self._generate_file_entries(path=kwargs.get('path', self.path), parent=self._gitems_parent)
        self.path = abspath(self.path)
        ev = self._create_files_entries_ev
        if ev is not None:
            ev.cancel()
        self._hide_progress()
        if self._create_files_entries():
            if ev is None:
                ev = self._create_files_entries_ev = Clock.schedule_interval(self._create_files_entries, 0.1)
            ev()

    def _get_file_paths(self, items):
        return [file.path for file in items]

    def _create_files_entries(self, *args):
        start = time()
        finished = False
        index = total = count = 1
        while time() - start < 0.05 or count < 10:
            try:
                index, total, item = next(self._gitems_gen)
                self._gitems.append(item)
                count += 1
            except StopIteration:
                finished = True
                break
            except TypeError:
                finished = True
                break
        if not finished:
            self._show_progress()
            self._progress.total = total
            self._progress.index = index
            return True
        self._items = items = self._gitems
        parent = self._gitems_parent
        if parent is None:
            self.dispatch('on_entries_cleared')
            for entry in items:
                self.dispatch('on_entry_added', entry, parent)
        else:
            parent.entries[:] = items
            for entry in items:
                self.dispatch('on_subentry_to_entry', entry, parent)
        self.files[:] = self._get_file_paths(items)
        self._hide_progress()
        self._gitems = None
        self._gitems_gen = None
        ev = self._create_files_entries_ev
        if ev is not None:
            ev.cancel()
        return False

    def cancel(self, *largs):
        """Cancel any background action started by filechooser, such as loading
        a new directory.

        .. versionadded:: 1.2.0
        """
        ev = self._create_files_entries_ev
        if ev is not None:
            ev.cancel()
        self._hide_progress()
        if len(self._previous_path) > 1:
            self.path = self._previous_path[-2]
            ev = self._update_files_ev
            if ev is not None:
                ev.cancel()

    def _show_progress(self):
        if self._progress:
            return
        cls = self.progress_cls
        if isinstance(cls, string_types):
            cls = Factory.get(cls)
        self._progress = cls(path=self.path)
        self._progress.value = 0
        self.add_widget(self._progress)

    def _hide_progress(self):
        if self._progress:
            self.remove_widget(self._progress)
            self._progress = None

    def _generate_file_entries(self, *args, **kwargs):
        is_root = False
        path = kwargs.get('path', self.path)
        have_parent = kwargs.get('parent', None) is not None
        if self.rootpath:
            rootpath = realpath(self.rootpath)
            path = realpath(path)
            if not path.startswith(rootpath):
                self.path = rootpath
                return
            elif path == rootpath:
                is_root = True
        elif platform == 'win':
            is_root = splitdrive(path)[1] in (sep, altsep)
        elif platform in ('macosx', 'linux', 'android', 'ios'):
            is_root = normpath(expanduser(path)) == sep
        else:
            Logger.warning('Filechooser: Unsupported OS: %r' % platform)
        if not is_root and (not have_parent):
            back = '..' + sep
            if platform == 'win':
                new_path = path[:path.rfind(sep)]
                if sep not in new_path:
                    new_path += sep
                pardir = self._create_entry_widget(dict(name=back, size='', path=new_path, controller=ref(self), isdir=True, parent=None, sep=sep, get_nice_size=lambda: ''))
            else:
                pardir = self._create_entry_widget(dict(name=back, size='', path=back, controller=ref(self), isdir=True, parent=None, sep=sep, get_nice_size=lambda: ''))
            yield (0, 1, pardir)
        try:
            for index, total, item in self._add_files(path):
                yield (index, total, item)
        except OSError:
            Logger.exception('Unable to open directory <%s>' % self.path)
            self.files[:] = []

    def _create_entry_widget(self, ctx):
        template = self.layout._ENTRY_TEMPLATE if self.layout else self._ENTRY_TEMPLATE
        return Builder.template(template, **ctx)

    def _add_files(self, path, parent=None):
        path = expanduser(path)
        if isfile(path):
            path = dirname(path)
        files = []
        fappend = files.append
        for f in self.file_system.listdir(path):
            try:
                fappend(normpath(join(path, f)))
            except UnicodeDecodeError:
                Logger.exception('unable to decode <{}>'.format(f))
            except UnicodeEncodeError:
                Logger.exception('unable to encode <{}>'.format(f))
        files = self._apply_filters(files)
        files = self.sort_func(files, self.file_system)
        is_hidden = self.file_system.is_hidden
        if not self.show_hidden:
            files = [x for x in files if not is_hidden(x)]
        self.files[:] = files
        total = len(files)
        wself = ref(self)
        for index, fn in enumerate(files):

            def get_nice_size():
                return self.get_nice_size(fn)
            ctx = {'name': basename(fn), 'get_nice_size': get_nice_size, 'path': fn, 'controller': wself, 'isdir': self.file_system.is_dir(fn), 'parent': parent, 'sep': sep}
            entry = self._create_entry_widget(ctx)
            yield (index, total, entry)

    def entry_subselect(self, entry):
        if not self.file_system.is_dir(entry.path):
            return
        self._update_files(path=entry.path, parent=entry)

    def close_subselection(self, entry):
        for subentry in entry.entries:
            self.dispatch('on_remove_subentry', subentry, entry)