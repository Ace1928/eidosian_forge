from os.path import join, exists
from os import getcwd
from collections import defaultdict
from kivy.core import core_select_lib
from kivy.clock import Clock
from kivy.config import Config
from kivy.logger import Logger
from kivy.base import EventLoop, stopTouchApp
from kivy.modules import Modules
from kivy.event import EventDispatcher
from kivy.properties import ListProperty, ObjectProperty, AliasProperty, \
from kivy.utils import platform, reify, deprecated, pi_version
from kivy.context import get_current_context
from kivy.uix.behaviors import FocusBehavior
from kivy.setupconfig import USE_SDL2
from kivy.graphics.transformation import Matrix
from kivy.graphics.cgl import cgl_get_backend_name
class WindowBase(EventDispatcher):
    """WindowBase is an abstract window widget for any window implementation.

    :Parameters:
        `borderless`: str, one of ('0', '1')
            Set the window border state. Check the
            :mod:`~kivy.config` documentation for a
            more detailed explanation on the values.
        `custom_titlebar`: str, one of ('0', '1')
            Set to `'1'` to uses a custom titlebar
        `fullscreen`: str, one of ('0', '1', 'auto', 'fake')
            Make the window fullscreen. Check the
            :mod:`~kivy.config` documentation for a
            more detailed explanation on the values.
        `width`: int
            Width of the window.
        `height`: int
            Height of the window.
        `minimum_width`: int
            Minimum width of the window (only works for sdl2 window provider).
        `minimum_height`: int
            Minimum height of the window (only works for sdl2 window provider).
        `always_on_top`: bool
            When enabled, the window will be brought to the front and will keep
            the window above the rest. If disabled, it will restore the default
            behavior. Only works for the sdl2 window provider.
        `allow_screensaver`: bool
            Allow the device to show a screen saver, or to go to sleep
            on mobile devices. Defaults to True. Only works for sdl2 window
            provider.

    :Events:
        `on_motion`: etype, motionevent
            Fired when a new :class:`~kivy.input.motionevent.MotionEvent` is
            dispatched
        `on_touch_down`:
            Fired when a new touch event is initiated.
        `on_touch_move`:
            Fired when an existing touch event changes location.
        `on_touch_up`:
            Fired when an existing touch event is terminated.
        `on_draw`:
            Fired when the :class:`Window` is being drawn.
        `on_flip`:
            Fired when the :class:`Window` GL surface is being flipped.
        `on_rotate`: rotation
            Fired when the :class:`Window` is being rotated.
        `on_close`:
            Fired when the :class:`Window` is closed.
        `on_request_close`:
            Fired when the event loop wants to close the window, or if the
            escape key is pressed and `exit_on_escape` is `True`. If a function
            bound to this event returns `True`, the window will not be closed.
            If the event is triggered because of the keyboard escape key,
            the keyword argument `source` is dispatched along with a value of
            `keyboard` to the bound functions.

            .. versionadded:: 1.9.0

        `on_cursor_enter`:
            Fired when the cursor enters the window.

            .. versionadded:: 1.9.1

        `on_cursor_leave`:
            Fired when the cursor leaves the window.

            .. versionadded:: 1.9.1

        `on_minimize`:
            Fired when the window is minimized.

            .. versionadded:: 1.10.0

        `on_maximize`:
            Fired when the window is maximized.

            .. versionadded:: 1.10.0

        `on_restore`:
            Fired when the window is restored.

            .. versionadded:: 1.10.0

        `on_hide`:
            Fired when the window is hidden.

            .. versionadded:: 1.10.0

        `on_show`:
            Fired when the window is shown.

            .. versionadded:: 1.10.0

        `on_keyboard`: key, scancode, codepoint, modifier
            Fired when the keyboard is used for input.

            .. versionchanged:: 1.3.0
                The *unicode* parameter has been deprecated in favor of
                codepoint, and will be removed completely in future versions.

        `on_key_down`: key, scancode, codepoint, modifier
            Fired when a key pressed.

            .. versionchanged:: 1.3.0
                The *unicode* parameter has been deprecated in favor of
                codepoint, and will be removed completely in future versions.

        `on_key_up`: key, scancode, codepoint
            Fired when a key is released.

            .. versionchanged:: 1.3.0
                The *unicode* parameter has be deprecated in favor of
                codepoint, and will be removed completely in future versions.

        `on_drop_begin`: x, y, *args
            Fired when text(s) or file(s) drop on the application is about to
            begin.

            .. versionadded:: 2.1.0

        `on_drop_file`: filename (bytes), x, y, *args
            Fired when a file is dropped on the application.

            .. versionadded:: 1.2.0

            .. versionchanged:: 2.1.0
                Renamed from `on_dropfile` to `on_drop_file`.

        `on_drop_text`: text (bytes), x, y, *args
            Fired when a text is dropped on the application.

            .. versionadded:: 2.1.0

        `on_drop_end`: x, y, *args
            Fired when text(s) or file(s) drop on the application has ended.

            .. versionadded:: 2.1.0

        `on_memorywarning`:
            Fired when the platform have memory issue (iOS / Android mostly)
            You can listen to this one, and clean whatever you can.

            .. versionadded:: 1.9.0

        `on_textedit(self, text)`:
            Fired when inputting with IME.
            The string inputting with IME is set as the parameter of
            this event.

            .. versionadded:: 1.10.1
    """
    __instance = None
    __initialized = False
    _fake_fullscreen = False
    _density = NumericProperty(1.0)
    _size = ListProperty([0, 0])
    _modifiers = ListProperty([])
    _rotation = NumericProperty(0)
    _focus = BooleanProperty(True)
    gl_backends_allowed = []
    '\n    A list of Kivy gl backend names, which if not empty, will be the\n    exclusive list of gl backends that can be used with this window.\n    '
    gl_backends_ignored = []
    '\n    A list of Kivy gl backend names that may not be used with this window.\n    '
    managed_textinput = False
    '\n    True if this Window class uses `on_textinput` to insert text, internal.\n    '
    children = ListProperty([])
    "List of the children of this window.\n\n    :attr:`children` is a :class:`~kivy.properties.ListProperty` instance and\n    defaults to an empty list.\n\n    Use :meth:`add_widget` and :meth:`remove_widget` to manipulate the list of\n    children. Don't manipulate the list directly unless you know what you are\n    doing.\n    "
    parent = ObjectProperty(None, allownone=True)
    'Parent of this window.\n\n    :attr:`parent` is a :class:`~kivy.properties.ObjectProperty` instance and\n    defaults to None. When created, the parent is set to the window itself.\n    You must take care of it if you are doing a recursive check.\n    '
    icon = StringProperty()
    'A path to the window icon.\n\n    .. versionadded:: 1.1.2\n\n    :attr:`icon` is a :class:`~kivy.properties.StringProperty`.\n    '

    def _get_modifiers(self):
        return self._modifiers
    modifiers = AliasProperty(_get_modifiers, None, bind=('_modifiers',))
    'List of keyboard modifiers currently active.\n\n    .. versionadded:: 1.0.9\n\n    :attr:`modifiers` is an :class:`~kivy.properties.AliasProperty`.\n    '

    def _get_size(self):
        r = self._rotation
        w, h = self._size
        if platform == 'win' or self._density != 1:
            w, h = self._win._get_gl_size()
        if self.softinput_mode == 'resize':
            h -= self.keyboard_height
        if r in (0, 180):
            return (w, h)
        return (h, w)

    def _set_size(self, size):
        if self._size != size:
            r = self._rotation
            if r in (0, 180):
                self._size = size
            else:
                self._size = (size[1], size[0])
            self.dispatch('on_pre_resize', *size)
    minimum_width = NumericProperty(0)
    'The minimum width to restrict the window to.\n\n    .. versionadded:: 1.9.1\n\n    :attr:`minimum_width` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 0.\n    '
    minimum_height = NumericProperty(0)
    'The minimum height to restrict the window to.\n\n    .. versionadded:: 1.9.1\n\n    :attr:`minimum_height` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 0.\n    '
    always_on_top = BooleanProperty(False)
    "When enabled, the window will be brought to the front and will keep\n    the window above the rest. If disabled, it will restore the default\n    behavior.\n\n    This option can be toggled freely during the window's lifecycle.\n\n    Only works for the sdl2 window provider. Check the :mod:`~kivy.config`\n    documentation for a more detailed explanation on the values.\n\n    .. versionadded:: 2.2.0\n\n    :attr:`always_on_top` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to False.\n    "
    allow_screensaver = BooleanProperty(True)
    'Whether the screen saver is enabled, or on mobile devices whether the\n    device is allowed to go to sleep while the app is open.\n\n    .. versionadded:: 1.10.0\n\n    :attr:`allow_screensaver` is a :class:`~kivy.properties.BooleanProperty`\n    and defaults to True.\n    '
    size = AliasProperty(_get_size, _set_size, bind=('_size', '_rotation'))
    'Get the rotated size of the window. If :attr:`rotation` is set, then the\n    size will change to reflect the rotation.\n\n    .. versionadded:: 1.0.9\n\n    :attr:`size` is an :class:`~kivy.properties.AliasProperty`.\n    '
    clearcolor = ColorProperty((0, 0, 0, 1))
    "Color used to clear the window.\n\n    ::\n\n        from kivy.core.window import Window\n\n        # red background color\n        Window.clearcolor = (1, 0, 0, 1)\n\n        # don't clear background at all\n        Window.clearcolor = None\n\n    .. versionchanged:: 1.7.2\n        The clearcolor default value is now: (0, 0, 0, 1).\n\n    .. versionadded:: 1.0.9\n\n    :attr:`clearcolor` is an :class:`~kivy.properties.ColorProperty` and\n    defaults to (0, 0, 0, 1).\n\n    .. versionchanged:: 2.1.0\n        Changed from :class:`~kivy.properties.AliasProperty` to\n        :class:`~kivy.properties.ColorProperty`.\n    "

    def _get_width(self):
        _size = self._size
        if platform == 'win' or self._density != 1:
            _size = self._win._get_gl_size()
        r = self._rotation
        if r == 0 or r == 180:
            return _size[0]
        return _size[1]
    width = AliasProperty(_get_width, bind=('_rotation', '_size', '_density'))
    'Rotated window width.\n\n    :attr:`width` is a read-only :class:`~kivy.properties.AliasProperty`.\n    '

    def _get_height(self):
        """Rotated window height"""
        r = self._rotation
        _size = self._size
        if platform == 'win' or self._density != 1:
            _size = self._win._get_gl_size()
        kb = self.keyboard_height if self.softinput_mode == 'resize' else 0
        if r == 0 or r == 180:
            return _size[1] - kb
        return _size[0] - kb
    height = AliasProperty(_get_height, bind=('_rotation', '_size', '_density'))
    'Rotated window height.\n\n    :attr:`height` is a read-only :class:`~kivy.properties.AliasProperty`.\n    '

    def _get_center(self):
        return (self.width / 2.0, self.height / 2.0)
    center = AliasProperty(_get_center, bind=('width', 'height'))
    'Center of the rotated window.\n\n    .. versionadded:: 1.0.9\n\n    :attr:`center` is an :class:`~kivy.properties.AliasProperty`.\n    '

    def _get_rotation(self):
        return self._rotation

    def _set_rotation(self, x):
        x = int(x % 360)
        if x == self._rotation:
            return
        if x not in (0, 90, 180, 270):
            raise ValueError('can rotate only 0, 90, 180, 270 degrees')
        self._rotation = x
        if not self.initialized:
            return
        self.dispatch('on_pre_resize', *self.size)
        self.dispatch('on_rotate', x)
    rotation = AliasProperty(_get_rotation, _set_rotation, bind=('_rotation',))
    'Get/set the window content rotation. Can be one of 0, 90, 180, 270\n    degrees.\n\n    .. versionadded:: 1.0.9\n\n    :attr:`rotation` is an :class:`~kivy.properties.AliasProperty`.\n    '
    softinput_mode = OptionProperty('', options=('', 'below_target', 'pan', 'scale', 'resize'))
    "This specifies the behavior of window contents on display of the soft\n    keyboard on mobile platforms. It can be one of '', 'pan', 'scale',\n    'resize' or 'below_target'. Their effects are listed below.\n\n    +----------------+-------------------------------------------------------+\n    | Value          | Effect                                                |\n    +================+=======================================================+\n    | ''             | The main window is left as is, allowing you to use    |\n    |                | the :attr:`keyboard_height` to manage the window      |\n    |                | contents manually.                                    |\n    +----------------+-------------------------------------------------------+\n    | 'pan'          | The main window pans, moving the bottom part of the   |\n    |                | window to be always on top of the keyboard.           |\n    +----------------+-------------------------------------------------------+\n    | 'resize'       | The window is resized and the contents scaled to fit  |\n    |                | the remaining space.                                  |\n    +----------------+-------------------------------------------------------+\n    | 'below_target' | The window pans so that the current target TextInput  |\n    |                | widget requesting the keyboard is presented just above|\n    |                | the soft keyboard.                                    |\n    +----------------+-------------------------------------------------------+\n\n    :attr:`softinput_mode` is an :class:`~kivy.properties.OptionProperty` and\n    defaults to ''.\n\n    .. note:: The `resize` option does not currently work with SDL2 on Android.\n\n    .. versionadded:: 1.9.0\n\n    .. versionchanged:: 1.9.1\n        The 'below_target' option was added.\n    "
    _keyboard_changed = BooleanProperty(False)
    _kheight = NumericProperty(0)
    _kanimation = None

    def _free_kanimation(self, *largs):
        WindowBase._kanimation = None

    def _animate_content(self):
        """Animate content to IME height.
        """
        kargs = self.keyboard_anim_args
        global Animation
        if not Animation:
            from kivy.animation import Animation
        if WindowBase._kanimation:
            WindowBase._kanimation.cancel(self)
        WindowBase._kanimation = kanim = Animation(_kheight=self.keyboard_height + self.keyboard_padding, d=kargs['d'], t=kargs['t'])
        kanim.bind(on_complete=self._free_kanimation)
        kanim.start(self)

    def _upd_kbd_height(self, *kargs):
        self._keyboard_changed = not self._keyboard_changed
        self._animate_content()

    def _get_ios_kheight(self):
        import ios
        return ios.get_kheight()

    def _get_android_kheight(self):
        if USE_SDL2:
            return 0
        global android
        if not android:
            import android
        return android.get_keyboard_height()

    def _get_kivy_vkheight(self):
        mode = Config.get('kivy', 'keyboard_mode')
        if mode in ['dock', 'systemanddock'] and self._vkeyboard_cls is not None:
            for w in self.children:
                if isinstance(w, self._vkeyboard_cls):
                    vkeyboard_height = w.height * w.scale
                    if self.softinput_mode == 'pan':
                        return vkeyboard_height
                    elif self.softinput_mode == 'below_target' and w.target.y < vkeyboard_height:
                        return vkeyboard_height - w.target.y
        return 0

    def _get_kheight(self):
        if platform == 'android':
            return self._get_android_kheight()
        elif platform == 'ios':
            return self._get_ios_kheight()
        return self._get_kivy_vkheight()
    keyboard_height = AliasProperty(_get_kheight, bind=('_keyboard_changed',))
    'Returns the height of the softkeyboard/IME on mobile platforms.\n    Will return 0 if not on mobile platform or if IME is not active.\n\n    .. note:: This property returns 0 with SDL2 on Android, but setting\n              Window.softinput_mode does work.\n\n    .. versionadded:: 1.9.0\n\n    :attr:`keyboard_height` is a read-only\n    :class:`~kivy.properties.AliasProperty` and defaults to 0.\n    '
    keyboard_anim_args = {'t': 'in_out_quart', 'd': 0.5}
    "The attributes for animating softkeyboard/IME.\n    `t` = `transition`, `d` = `duration`. This value will have no effect on\n    desktops.\n\n    .. versionadded:: 1.10.0\n\n    :attr:`keyboard_anim_args` is a dict and defaults to\n    {'t': 'in_out_quart', 'd': `.5`}.\n    "
    keyboard_padding = NumericProperty(0)
    'The padding to have between the softkeyboard/IME & target\n    or bottom of window. Will have no effect on desktops.\n\n    .. versionadded:: 1.10.0\n\n    :attr:`keyboard_padding` is a\n    :class:`~kivy.properties.NumericProperty` and defaults to 0.\n    '

    def _set_system_size(self, size):
        self._size = size

    def _get_system_size(self):
        if self.softinput_mode == 'resize':
            return (self._size[0], self._size[1] - self.keyboard_height)
        return self._size
    system_size = AliasProperty(_get_system_size, _set_system_size, bind=('_size',))
    'Real size of the window ignoring rotation. If the density is\n    not 1, the :attr:`system_size` is the :attr:`size` divided by\n    density.\n\n    .. versionadded:: 1.0.9\n\n    :attr:`system_size` is an :class:`~kivy.properties.AliasProperty`.\n    '

    def _get_effective_size(self):
        """On density=1 and non-ios / non-Windows displays,
        return :attr:`system_size`, else return scaled / rotated :attr:`size`.

        Used by MouseMotionEvent.update_graphics() and WindowBase.on_motion().
        """
        w, h = self.system_size
        if platform in ('ios', 'win') or self._density != 1:
            w, h = self.size
        return (w, h)
    borderless = BooleanProperty(False)
    'When set to True, this property removes the window border/decoration.\n    Check the :mod:`~kivy.config` documentation for a more detailed\n    explanation on the values.\n\n    .. versionadded:: 1.9.0\n\n    :attr:`borderless` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to False.\n    '
    custom_titlebar = BooleanProperty(False)
    'When set to True, allows the user to set a widget as a titlebar.\n    Check the :mod:`~kivy.config` documentation for a more detailed\n    explanation on the values.\n\n    .. versionadded:: 2.1.0\n\n    see :meth:`~kivy.core.window.WindowBase.set_custom_titlebar`\n    for detailed usage\n    :attr:`custom_titlebar` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to False.\n    '
    fullscreen = OptionProperty(False, options=(True, False, 'auto', 'fake'))
    "This property sets the fullscreen mode of the window. Available options\n    are: True, False, 'auto' and 'fake'. Check the :mod:`~kivy.config`\n    documentation for more detailed explanations on these values.\n\n    fullscreen is an :class:`~kivy.properties.OptionProperty` and defaults to\n    `False`.\n\n    .. versionadded:: 1.2.0\n\n    .. note::\n        The 'fake' option has been deprecated, use the :attr:`borderless`\n        property instead.\n\n    .. warning::\n        On iOS, setting :attr:`fullscreen` to `False` will not automatically\n        hide the status bar.\n\n        To achieve this, you must set :attr:`fullscreen` to `False`, and\n        then also set :attr:`borderless` to `False`.\n    "
    mouse_pos = ObjectProperty((0, 0))
    '2d position of the mouse cursor within the window.\n\n    Position is relative to the left/bottom point of the window.\n\n    .. note::\n        Cursor position will be scaled by the pixel density if the high density\n        mode is supported by the window provider.\n\n    .. versionadded:: 1.2.0\n\n    :attr:`mouse_pos` is an :class:`~kivy.properties.ObjectProperty` and\n    defaults to (0, 0).\n    '
    show_cursor = BooleanProperty(True)
    'Set whether or not the cursor is shown on the window.\n\n    .. versionadded:: 1.9.1\n\n    :attr:`show_cursor` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to True.\n    '

    def _get_focus(self):
        return self._focus
    focus = AliasProperty(_get_focus, None, bind=('_focus',))
    'Check whether or not the window currently has focus.\n\n    .. versionadded:: 1.9.1\n\n    :attr:`focus` is a read-only :class:`~kivy.properties.AliasProperty` and\n    defaults to True.\n    '

    def _set_cursor_state(self, value):
        pass

    def set_system_cursor(self, cursor_name):
        """Set type of a mouse cursor in the Window.

        It can be one of 'arrow', 'ibeam', 'wait', 'crosshair', 'wait_arrow',
        'size_nwse', 'size_nesw', 'size_we', 'size_ns', 'size_all', 'no', or
        'hand'.

        On some platforms there might not be a specific cursor supported and
        such an option falls back to one of the substitutable alternatives:

        +------------+-----------+------------+-----------+---------------+
        |            | Windows   | MacOS      | Linux X11 | Linux Wayland |
        +============+===========+============+===========+===============+
        | arrow      | arrow     | arrow      | arrow     | arrow         |
        +------------+-----------+------------+-----------+---------------+
        | ibeam      | ibeam     | ibeam      | ibeam     | ibeam         |
        +------------+-----------+------------+-----------+---------------+
        | wait       | wait      | arrow      | wait      | wait          |
        +------------+-----------+------------+-----------+---------------+
        | crosshair  | crosshair | crosshair  | crosshair | hand          |
        +------------+-----------+------------+-----------+---------------+
        | wait_arrow | arrow     | arrow      | wait      | wait          |
        +------------+-----------+------------+-----------+---------------+
        | size_nwse  | size_nwse | size_all   | size_all  | hand          |
        +------------+-----------+------------+-----------+---------------+
        | size_nesw  | size_nesw | size_all   | size_all  | hand          |
        +------------+-----------+------------+-----------+---------------+
        | size_we    | size_we   | size_we    | size_we   | hand          |
        +------------+-----------+------------+-----------+---------------+
        | size_ns    | size_ns   | size_ns    | size_ns   | hand          |
        +------------+-----------+------------+-----------+---------------+
        | size_all   | size_all  | size_all   | size_all  | hand          |
        +------------+-----------+------------+-----------+---------------+
        | no         | no        | no         | no        | ibeam         |
        +------------+-----------+------------+-----------+---------------+
        | hand       | hand      | hand       | hand      | hand          |
        +------------+-----------+------------+-----------+---------------+

        .. versionadded:: 1.10.1

        .. note::
            This feature requires the SDL2 window provider and is currently
            only supported on desktop platforms.
        """
        pass

    def _get_window_pos(self):
        pass

    def _set_window_pos(self, x, y):
        pass

    def _get_left(self):
        if not self.initialized:
            return self._left
        return self._get_window_pos()[0]

    def _set_left(self, value):
        pos = self._get_window_pos()
        self._set_window_pos(value, pos[1])

    def _get_top(self):
        if not self.initialized:
            return self._top
        return self._get_window_pos()[1]

    def _set_top(self, value):
        pos = self._get_window_pos()
        self._set_window_pos(pos[0], value)
    top = AliasProperty(_get_top, _set_top)
    "Top position of the window.\n\n    .. note:: It's an SDL2 property with `[0, 0]` in the top-left corner.\n\n    .. versionchanged:: 1.10.0\n        :attr:`top` is now an :class:`~kivy.properties.AliasProperty`\n\n    .. versionadded:: 1.9.1\n\n    :attr:`top` is an :class:`~kivy.properties.AliasProperty` and defaults to\n    the position set in :class:`~kivy.config.Config`.\n    "
    left = AliasProperty(_get_left, _set_left)
    "Left position of the window.\n\n    .. note:: It's an SDL2 property with `[0, 0]` in the top-left corner.\n\n    .. versionchanged:: 1.10.0\n        :attr:`left` is now an :class:`~kivy.properties.AliasProperty`\n\n    .. versionadded:: 1.9.1\n\n    :attr:`left` is an :class:`~kivy.properties.AliasProperty` and defaults to\n    the position set in :class:`~kivy.config.Config`.\n    "

    def _get_opacity(self):
        return self._get_window_opacity()

    def _set_opacity(self, opacity):
        return self._set_window_opacity(opacity)

    def _get_window_opacity(self):
        Logger.warning('Window: Opacity is not implemented in the current window provider')

    def _set_window_opacity(self, opacity):
        Logger.warning('Window: Opacity is not implemented in the current window provider')
    opacity = AliasProperty(_get_opacity, _set_opacity, cache=True)
    'Opacity of the window. Accepts a value between 0.0 (transparent) and\n    1.0 (opaque).\n\n    .. note::\n        This feature requires the SDL2 window provider.\n\n    .. versionadded:: 2.3.0\n\n    :attr:`opacity` is an :class:`~kivy.properties.AliasProperty` and defaults\n    to `1.0`.\n    '

    @property
    def __self__(self):
        return self
    position = OptionProperty('auto', options=['auto', 'custom'])
    render_context = ObjectProperty(None)
    canvas = ObjectProperty(None)
    title = StringProperty('Kivy')
    event_managers = None
    "Holds a `list` of registered event managers.\n\n    Don't change the property directly but use\n    :meth:`register_event_manager` and :meth:`unregister_event_manager` to\n    register and unregister an event manager.\n\n    Event manager is an instance of\n    :class:`~kivy.eventmanager.EventManagerBase`.\n\n    .. versionadded:: 2.1.0\n\n    .. warning::\n        This is an experimental property and it remains so while this warning\n        is present.\n    "
    event_managers_dict = None
    "Holds a `dict` of `type_id` to `list` of event managers.\n\n    Don't change the property directly but use\n    :meth:`register_event_manager` and :meth:`unregister_event_manager` to\n    register and unregister an event manager.\n\n    Event manager is an instance of\n    :class:`~kivy.eventmanager.EventManagerBase`.\n\n    .. versionadded:: 2.1.0\n\n    .. warning::\n        This is an experimental property and it remains so while this warning\n        is present.\n    "
    trigger_create_window = None
    __events__ = ('on_draw', 'on_flip', 'on_rotate', 'on_resize', 'on_move', 'on_close', 'on_minimize', 'on_maximize', 'on_restore', 'on_hide', 'on_show', 'on_motion', 'on_touch_down', 'on_touch_move', 'on_touch_up', 'on_mouse_down', 'on_mouse_move', 'on_mouse_up', 'on_keyboard', 'on_key_down', 'on_key_up', 'on_textinput', 'on_drop_begin', 'on_drop_file', 'on_dropfile', 'on_drop_text', 'on_drop_end', 'on_request_close', 'on_cursor_enter', 'on_cursor_leave', 'on_joy_axis', 'on_joy_hat', 'on_joy_ball', 'on_joy_button_down', 'on_joy_button_up', 'on_memorywarning', 'on_textedit', 'on_pre_resize')

    def __new__(cls, **kwargs):
        if cls.__instance is None:
            cls.__instance = EventDispatcher.__new__(cls)
        return cls.__instance

    def __init__(self, **kwargs):
        force = kwargs.pop('force', False)
        if WindowBase.__instance is not None and (not force):
            return
        self.initialized = False
        self.event_managers = []
        self.event_managers_dict = defaultdict(list)
        self._is_desktop = Config.getboolean('kivy', 'desktop')
        self.trigger_create_window = Clock.create_trigger(self.create_window, -1)
        self.trigger_keyboard_height = Clock.create_trigger(self._upd_kbd_height, 0.5)
        self.bind(_kheight=lambda *args: self.update_viewport())
        if 'borderless' not in kwargs:
            kwargs['borderless'] = Config.getboolean('graphics', 'borderless')
        if 'custom_titlebar' not in kwargs:
            kwargs['custom_titlebar'] = Config.getboolean('graphics', 'custom_titlebar')
        if 'fullscreen' not in kwargs:
            fullscreen = Config.get('graphics', 'fullscreen')
            if fullscreen not in ('auto', 'fake'):
                fullscreen = fullscreen.lower() in ('true', '1', 'yes')
            kwargs['fullscreen'] = fullscreen
        if 'width' not in kwargs:
            kwargs['width'] = Config.getint('graphics', 'width')
        if 'height' not in kwargs:
            kwargs['height'] = Config.getint('graphics', 'height')
        if 'minimum_width' not in kwargs:
            kwargs['minimum_width'] = Config.getint('graphics', 'minimum_width')
        if 'minimum_height' not in kwargs:
            kwargs['minimum_height'] = Config.getint('graphics', 'minimum_height')
        if 'always_on_top' not in kwargs:
            kwargs['always_on_top'] = Config.getboolean('graphics', 'always_on_top')
        if 'allow_screensaver' not in kwargs:
            kwargs['allow_screensaver'] = Config.getboolean('graphics', 'allow_screensaver')
        if 'rotation' not in kwargs:
            kwargs['rotation'] = Config.getint('graphics', 'rotation')
        if 'position' not in kwargs:
            kwargs['position'] = Config.getdefault('graphics', 'position', 'auto')
        if 'top' in kwargs:
            kwargs['position'] = 'custom'
            self._top = kwargs['top']
        else:
            self._top = Config.getint('graphics', 'top')
        if 'left' in kwargs:
            kwargs['position'] = 'custom'
            self._left = kwargs['left']
        else:
            self._left = Config.getint('graphics', 'left')
        kwargs['_size'] = (kwargs.pop('width'), kwargs.pop('height'))
        if 'show_cursor' not in kwargs:
            kwargs['show_cursor'] = Config.getboolean('graphics', 'show_cursor')
        if 'shape_image' not in kwargs:
            kwargs['shape_image'] = Config.get('kivy', 'window_shape')
        self.fbind('on_drop_file', lambda win, filename, *args: win.dispatch('on_dropfile', filename))
        super(WindowBase, self).__init__(**kwargs)
        self._bind_create_window()
        self.bind(size=self.trigger_keyboard_height, rotation=self.trigger_keyboard_height)
        self.bind(softinput_mode=lambda *dt: self.update_viewport(), keyboard_height=lambda *dt: self.update_viewport())
        self.bind(show_cursor=lambda *dt: self._set_cursor_state(dt[1]))
        self._system_keyboard = Keyboard(window=self)
        self._keyboards = {'system': self._system_keyboard}
        self._vkeyboard_cls = None
        self.children = []
        self.parent = self
        import kivy.core.gl
        self.create_window()
        self.register()
        self.configure_keyboards()
        if not hasattr(self, '_context'):
            self._context = get_current_context()
        self.fbind('dpi', self._reset_metrics_dpi)
        self.initialized = True

    def _reset_metrics_dpi(self, *args):
        from kivy.metrics import Metrics
        Metrics.reset_dpi()

    def _bind_create_window(self):
        for prop in ('fullscreen', 'borderless', 'position', 'top', 'left', '_size', 'system_size'):
            self.bind(**{prop: self.trigger_create_window})

    def _unbind_create_window(self):
        for prop in ('fullscreen', 'borderless', 'position', 'top', 'left', '_size', 'system_size'):
            self.unbind(**{prop: self.trigger_create_window})

    def register(self):
        if self.initialized:
            return
        EventLoop.set_window(self)
        Modules.register_window(self)
        EventLoop.add_event_listener(self)

    def register_event_manager(self, manager):
        """Register and start an event manager to handle events declared in
        :attr:`~kivy.eventmanager.EventManagerBase.type_ids` attribute.

        .. versionadded:: 2.1.0

        .. warning::
            This is an experimental method and it remains so until this warning
            is present as it can be changed or removed in the next versions of
            Kivy.
        """
        self.event_managers.insert(0, manager)
        for type_id in manager.type_ids:
            self.event_managers_dict[type_id].insert(0, manager)
        manager.window = self
        manager.start()

    def unregister_event_manager(self, manager):
        """Unregister and stop an event manager previously registered with
        :meth:`register_event_manager`.

        .. versionadded:: 2.1.0

        .. warning::
            This is an experimental method and it remains so until this warning
            is present as it can be changed or removed in the next versions of
            Kivy.
        """
        self.event_managers.remove(manager)
        for type_id in manager.type_ids:
            self.event_managers_dict[type_id].remove(manager)
        manager.stop()
        manager.window = None

    def mainloop(self):
        """Called by the EventLoop every frame after it idles.
        """
        pass

    def maximize(self):
        """Maximizes the window. This method should be used on desktop
        platforms only.

        .. versionadded:: 1.9.0

        .. note::
            This feature requires the SDL2 window provider and is currently
            only supported on desktop platforms.
        """
        Logger.warning('Window: maximize() is not implemented in the current window provider.')

    def minimize(self):
        """Minimizes the window. This method should be used on desktop
        platforms only.

        .. versionadded:: 1.9.0

        .. note::
            This feature requires the SDL2 window provider and is currently
            only supported on desktop platforms.
        """
        Logger.warning('Window: minimize() is not implemented in the current window provider.')

    def restore(self):
        """Restores the size and position of a maximized or minimized window.
        This method should be used on desktop platforms only.

        .. versionadded:: 1.9.0

        .. note::
            This feature requires the SDL2 window provider and is currently
            only supported on desktop platforms.
        """
        Logger.warning('Window: restore() is not implemented in the current window provider.')

    def hide(self):
        """Hides the window. This method should be used on desktop
        platforms only.

        .. versionadded:: 1.9.0

        .. note::
            This feature requires the SDL2 window provider and is currently
            only supported on desktop platforms.
        """
        Logger.warning('Window: hide() is not implemented in the current window provider.')

    def show(self):
        """Shows the window. This method should be used on desktop
        platforms only.

        .. versionadded:: 1.9.0

        .. note::
            This feature requires the SDL2 window provider and is currently
            only supported on desktop platforms.
        """
        Logger.warning('Window: show() is not implemented in the current window provider.')

    def raise_window(self):
        """Raise the window. This method should be used on desktop
        platforms only.

        .. versionadded:: 1.9.1

        .. note::
            This feature requires the SDL2 window provider and is currently
            only supported on desktop platforms.
        """
        Logger.warning('Window: raise_window is not implemented in the current window provider.')

    def close(self):
        """Close the window"""
        self.dispatch('on_close')
        from kivy.cache import Cache
        from kivy.graphics.context import get_context
        Cache.remove('kv.loader')
        Cache.remove('kv.image')
        Cache.remove('kv.shader')
        Cache.remove('kv.texture')
        get_context().flush()
    shape_image = StringProperty('')
    "An image for the window shape (only works for sdl2 window provider).\n\n    .. warning:: The image size has to be the same like the window's size!\n\n    .. versionadded:: 1.10.1\n\n    :attr:`shape_image` is a :class:`~kivy.properties.StringProperty` and\n    defaults to 'data/images/defaultshape.png'. This value is taken from\n    :class:`~kivy.config.Config`.\n    "

    def set_custom_titlebar(self, widget):
        """
        Sets a Widget as a titlebar

            :widget: The widget you want to set as the titlebar

        .. versionadded:: 2.1.0

        This function returns `True` on successfully setting the custom titlebar,
        else false

        How to use this feature

        ::

            1. first set Window.custom_titlebar to True
            2. then call Window.set_custom_titlebar with the widget/layout you want to set as titlebar as the argument # noqa: E501

        If you want a child of the widget to receive touch events, in
        that child define a property `draggable` and set it to False

        If you set the property `draggable` on a layout,
        all the child in the layout will receive touch events

        If you want to override default behavior, add function `in_drag_area(x,y)`
        to the widget

        The function is call with two args x,y which are mouse.x, and mouse.y
        the function should return

        | `True` if that point should be used to drag the window
        | `False` if you want to receive the touch event at the point

        .. note::
            If you use :meth:`in_drag_area` property `draggable`
            will not be checked

        .. note::
            This feature requires the SDL2 window provider and is currently
            only supported on desktop platforms.

        .. warning::
            :mod:`~kivy.core.window.WindowBase.custom_titlebar` must be set to True
            for the widget to be successfully set as a titlebar

        """
        Logger.warning('Window: set_custom_titlebar is not implemented in the current window provider.')

    def on_shape_image(self, instance, value):
        if self.initialized:
            self._set_shape(shape_image=value, mode=self.shape_mode, cutoff=self.shape_cutoff, color_key=self.shape_color_key)
    shape_cutoff = BooleanProperty(True)
    'The window :attr:`shape_image` cutoff property (only works for sdl2\n    window provider).\n\n    .. versionadded:: 1.10.1\n\n    :attr:`shape_cutoff` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to True.\n    '

    def on_shape_cutoff(self, instance, value):
        self._set_shape(shape_image=self.shape_image, mode=self.shape_mode, cutoff=value, color_key=self.shape_color_key)

    def _get_shaped(self):
        return self._is_shaped()
    shaped = AliasProperty(_get_shaped, None)
    'Read only property to check if the window is shapable or not (only works\n    for sdl2 window provider).\n\n    .. versionadded:: 1.10.1\n\n    :attr:`shaped` is an :class:`~kivy.properties.AliasProperty`.\n    '

    def _get_shape_mode(self):
        if not self.shaped:
            return ''
        i = self._get_shaped_mode()['mode']
        modes = ('default', 'binalpha', 'reversebinalpha', 'colorkey')
        return modes[i]

    def _set_shape_mode(self, value):
        self._set_shaped_mode(value)
    shape_mode = AliasProperty(_get_shape_mode, _set_shape_mode)
    "Window mode for shaping (only works for sdl2 window provider).\n\n    - can be RGB only\n       - `default` - does nothing special\n       - `colorkey` - hides a color of the :attr:`shape_color_key`\n    - has to contain alpha channel\n       - `binalpha` - hides an alpha channel of the :attr:`shape_image`\n       - `reversebinalpha` - shows only the alpha of the :attr:`shape_image`\n\n    .. note::\n        Before actually setting the mode make sure the Window has the same\n        size like the :attr:`shape_image`, preferably via Config before\n        the Window is actually created.\n\n        If the :attr:`shape_image` isn't set, the default one will be used\n        and the mode might not take the desired visual effect.\n\n    .. versionadded:: 1.10.1\n\n    :attr:`shape_mode` is an :class:`~kivy.properties.AliasProperty`.\n    "
    shape_color_key = ColorProperty([1, 1, 1, 1])
    'Color key of the shaped window - sets which color will be hidden from\n    the window :attr:`shape_image` (only works for sdl2 window provider).\n\n    .. versionadded:: 1.10.1\n\n    :attr:`shape_color_key` is a :class:`~kivy.properties.ColorProperty`\n    instance and defaults to [1, 1, 1, 1].\n\n    .. versionchanged:: 2.0.0\n        Changed from :class:`~kivy.properties.ListProperty` to\n        :class:`~kivy.properties.ColorProperty`.\n    '

    def on_shape_color_key(self, instance, value):
        self._set_shape(shape_image=self.shape_image, mode=self.shape_mode, cutoff=self.shape_cutoff, color_key=value)

    def get_gl_backend_name(self):
        """
        Returns the gl backend that will or is used with this window.
        """
        return cgl_get_backend_name(allowed=self.gl_backends_allowed, ignored=self.gl_backends_ignored)

    def initialize_gl(self):
        from kivy.core.gl import init_gl
        init_gl(allowed=self.gl_backends_allowed, ignored=self.gl_backends_ignored)

    def create_window(self, *largs):
        """Will create the main window and configure it.

        .. warning::
            This method is called automatically at runtime. If you call it, it
            will recreate a RenderContext and Canvas. This means you'll have a
            new graphics tree, and the old one will be unusable.

            This method exist to permit the creation of a new OpenGL context
            AFTER closing the first one. (Like using runTouchApp() and
            stopTouchApp()).

            This method has only been tested in a unittest environment and
            is not suitable for Applications.

            Again, don't use this method unless you know exactly what you are
            doing!
        """
        self.trigger_create_window.cancel()
        if platform in 'android':
            self._unbind_create_window()
        if not self.initialized:
            self.initialize_gl()
            from kivy.graphics import RenderContext, Canvas
            self.render_context = RenderContext()
            self.canvas = Canvas()
            self.render_context.add(self.canvas)
        elif platform == 'linux' or Window.__class__.__name__ == 'WindowSDL':
            self.dispatch('on_pre_resize', *self.size)
        else:
            from kivy.graphics.context import get_context
            get_context().reload()
            Clock.schedule_once(lambda x: self.canvas.ask_update(), 0)
            self.dispatch('on_pre_resize', *self.size)
        self.update_viewport()

    def on_flip(self):
        """Flip between buffers (event)"""
        self.flip()

    def flip(self):
        """Flip between buffers"""
        pass

    def _update_childsize(self, instance, value):
        self.update_childsize([instance])

    def add_widget(self, widget, canvas=None):
        """Add a widget to a window"""
        if widget.parent:
            from kivy.uix.widget import WidgetException
            raise WidgetException('Cannot add %r to window, it already has a parent %r' % (widget, widget.parent))
        widget.parent = self
        self.children.insert(0, widget)
        canvas = self.canvas.before if canvas == 'before' else self.canvas.after if canvas == 'after' else self.canvas
        canvas.add(widget.canvas)
        self.update_childsize([widget])
        widget.bind(pos_hint=self._update_childsize, size_hint=self._update_childsize, size_hint_max=self._update_childsize, size_hint_min=self._update_childsize, size=self._update_childsize, pos=self._update_childsize)

    def remove_widget(self, widget):
        """Remove a widget from a window
        """
        if widget not in self.children:
            return
        self.children.remove(widget)
        if widget.canvas in self.canvas.children:
            self.canvas.remove(widget.canvas)
        elif widget.canvas in self.canvas.after.children:
            self.canvas.after.remove(widget.canvas)
        elif widget.canvas in self.canvas.before.children:
            self.canvas.before.remove(widget.canvas)
        widget.parent = None
        widget.unbind(pos_hint=self._update_childsize, size_hint=self._update_childsize, size_hint_max=self._update_childsize, size_hint_min=self._update_childsize, size=self._update_childsize, pos=self._update_childsize)

    def clear(self):
        """Clear the window with the background color"""
        from kivy.graphics import opengl as gl
        gl.glClearColor(*self.clearcolor)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT | gl.GL_STENCIL_BUFFER_BIT)

    def set_title(self, title):
        """Set the window title.

        .. versionadded:: 1.0.5
        """
        self.title = title

    def set_icon(self, filename):
        """Set the icon of the window.

        .. versionadded:: 1.0.5
        """
        self.icon = filename

    def to_widget(self, x, y, initial=True, relative=False):
        return (x, y)

    def to_window(self, x, y, initial=True, relative=False):
        return (x, y)

    def to_normalized_pos(self, x, y):
        """Transforms absolute coordinates to normalized (0-1) coordinates
        using :attr:`system_size`.

        .. versionadded:: 2.1.0
        """
        x_max = self.system_size[0] - 1.0
        y_max = self.system_size[1] - 1.0
        return (x / x_max if x_max > 0 else 0.0, y / y_max if y_max > 0 else 0.0)

    def transform_motion_event_2d(self, me, widget=None):
        """Transforms the motion event `me` to this window size and then if
        `widget` is passed transforms `me` to `widget`'s local coordinates.

        :raises:
            `AttributeError`: If widget's ancestor is ``None``.

        .. note::
            Unless it's a specific case, call
            :meth:`~kivy.input.motionevent.MotionEvent.push` before and
            :meth:`~kivy.input.motionevent.MotionEvent.pop` after this method's
            call to preserve previous values of `me`'s attributes.

        .. versionadded:: 2.1.0
        """
        width, height = self._get_effective_size()
        me.scale_for_screen(width, height, rotation=self.rotation, smode=self.softinput_mode, kheight=self.keyboard_height)
        if widget is not None:
            parent = widget.parent
            try:
                if parent:
                    me.apply_transform_2d(parent.to_widget)
                else:
                    me.apply_transform_2d(widget.to_widget)
                    me.apply_transform_2d(widget.to_parent)
            except AttributeError:
                raise

    def _apply_transform(self, m):
        return m

    def get_window_matrix(self, x=0, y=0):
        m = Matrix()
        m.translate(x, y, 0)
        return m

    def get_root_window(self):
        return self

    def get_parent_window(self):
        return self

    def get_parent_layout(self):
        return None

    def on_draw(self):
        self.clear()
        self.render_context.draw()

    def on_motion(self, etype, me):
        """Event called when a motion event is received.

        :Parameters:
            `etype`: str
                One of "begin", "update" or "end".
            `me`: :class:`~kivy.input.motionevent.MotionEvent`
                The motion event currently dispatched.

        .. versionchanged:: 2.1.0
            Event managers get to handle the touch event first and if none of
            them accepts the event (by returning `True`) then window will
            dispatch `me` through "on_touch_down", "on_touch_move",
            "on_touch_up" events depending on the `etype`. All non-touch events
            will go only through managers.
        """
        accepted = False
        for manager in self.event_managers_dict[me.type_id][:]:
            accepted = manager.dispatch(etype, me) or accepted
        if accepted:
            if me.is_touch and etype == 'end':
                FocusBehavior._handle_post_on_touch_up(me)
            return accepted
        if me.is_touch:
            self.transform_motion_event_2d(me)
            if etype == 'begin':
                self.dispatch('on_touch_down', me)
            elif etype == 'update':
                self.dispatch('on_touch_move', me)
            elif etype == 'end':
                self.dispatch('on_touch_up', me)
                FocusBehavior._handle_post_on_touch_up(me)

    def on_touch_down(self, touch):
        """Event called when a touch down event is initiated.

        .. versionchanged:: 1.9.0
            The touch `pos` is now transformed to window coordinates before
            this method is called. Before, the touch `pos` coordinate would be
            `(0, 0)` when this method was called.
        """
        for w in self.children[:]:
            if w.dispatch('on_touch_down', touch):
                return True

    def on_touch_move(self, touch):
        """Event called when a touch event moves (changes location).

        .. versionchanged:: 1.9.0
            The touch `pos` is now transformed to window coordinates before
            this method is called. Before, the touch `pos` coordinate would be
            `(0, 0)` when this method was called.
        """
        for w in self.children[:]:
            if w.dispatch('on_touch_move', touch):
                return True

    def on_touch_up(self, touch):
        """Event called when a touch event is released (terminated).

        .. versionchanged:: 1.9.0
            The touch `pos` is now transformed to window coordinates before
            this method is called. Before, the touch `pos` coordinate would be
            `(0, 0)` when this method was called.
        """
        for w in self.children[:]:
            if w.dispatch('on_touch_up', touch):
                return True

    def on_pre_resize(self, width, height):
        key = (width, height)
        if hasattr(self, '_last_resize') and self._last_resize == key:
            return
        self._last_resize = key
        self.dispatch('on_resize', width, height)

    def on_resize(self, width, height):
        """Event called when the window is resized."""
        self.update_viewport()

    def on_move(self):
        self.property('top').dispatch(self)
        self.property('left').dispatch(self)

    def update_viewport(self):
        from kivy.graphics.opengl import glViewport
        from kivy.graphics.transformation import Matrix
        from math import radians
        w, h = self._get_effective_size()
        smode = self.softinput_mode
        target = self._system_keyboard.target
        targettop = max(0, target.to_window(0, target.y)[1]) if target else 0
        kheight = self._kheight
        w2, h2 = (w / 2.0, h / 2.0)
        r = radians(self.rotation)
        y = 0
        _h = h
        if smode == 'pan':
            y = kheight
        elif smode == 'below_target':
            y = 0 if kheight < targettop else kheight - targettop
        if smode == 'scale':
            _h -= kheight
        glViewport(0, 0, w, _h)
        projection_mat = Matrix()
        projection_mat.view_clip(0.0, w, 0.0, h, -1.0, 1.0, 0)
        self.render_context['projection_mat'] = projection_mat
        modelview_mat = Matrix().translate(w2, h2, 0)
        modelview_mat = modelview_mat.multiply(Matrix().rotate(r, 0, 0, 1))
        w, h = self.size
        w2, h2 = (w / 2.0, h / 2.0 - y)
        modelview_mat = modelview_mat.multiply(Matrix().translate(-w2, -h2, 0))
        self.render_context['modelview_mat'] = modelview_mat
        frag_modelview_mat = Matrix()
        frag_modelview_mat.set(flat=modelview_mat.get())
        self.render_context['frag_modelview_mat'] = frag_modelview_mat
        self.canvas.ask_update()
        self.update_childsize()

    def update_childsize(self, childs=None):
        width, height = self.size
        if childs is None:
            childs = self.children
        for w in childs:
            shw, shh = w.size_hint
            shw_min, shh_min = w.size_hint_min
            shw_max, shh_max = w.size_hint_max
            if shw is not None and shh is not None:
                c_w = shw * width
                c_h = shh * height
                if shw_min is not None and c_w < shw_min:
                    c_w = shw_min
                elif shw_max is not None and c_w > shw_max:
                    c_w = shw_max
                if shh_min is not None and c_h < shh_min:
                    c_h = shh_min
                elif shh_max is not None and c_h > shh_max:
                    c_h = shh_max
                w.size = (c_w, c_h)
            elif shw is not None:
                c_w = shw * width
                if shw_min is not None and c_w < shw_min:
                    c_w = shw_min
                elif shw_max is not None and c_w > shw_max:
                    c_w = shw_max
                w.width = c_w
            elif shh is not None:
                c_h = shh * height
                if shh_min is not None and c_h < shh_min:
                    c_h = shh_min
                elif shh_max is not None and c_h > shh_max:
                    c_h = shh_max
                w.height = c_h
            for key, value in w.pos_hint.items():
                if key == 'x':
                    w.x = value * width
                elif key == 'right':
                    w.right = value * width
                elif key == 'y':
                    w.y = value * height
                elif key == 'top':
                    w.top = value * height
                elif key == 'center_x':
                    w.center_x = value * width
                elif key == 'center_y':
                    w.center_y = value * height

    def screenshot(self, name='screenshot{:04d}.png'):
        """Save the actual displayed image to a file.
        """
        i = 0
        path = None
        if name != 'screenshot{:04d}.png':
            _ext = name.split('.')[-1]
            name = ''.join((name[:-(len(_ext) + 1)], '{:04d}.', _ext))
        while True:
            i += 1
            path = join(getcwd(), name.format(i))
            if not exists(path):
                break
        return path

    def on_rotate(self, rotation):
        """Event called when the screen has been rotated.
        """
        pass

    def on_close(self, *largs):
        """Event called when the window is closed."""
        Modules.unregister_window(self)
        EventLoop.remove_event_listener(self)

    def on_minimize(self, *largs):
        """Event called when the window is minimized.

        .. versionadded:: 1.10.0

        .. note::
            This feature requires the SDL2 window provider.
        """
        pass

    def on_maximize(self, *largs):
        """Event called when the window is maximized.

        .. versionadded:: 1.10.0

        .. note::
            This feature requires the SDL2 window provider.
        """
        pass

    def on_restore(self, *largs):
        """Event called when the window is restored.

        .. versionadded:: 1.10.0

        .. note::
            This feature requires the SDL2 window provider.
        """
        pass

    def on_hide(self, *largs):
        """Event called when the window is hidden.

        .. versionadded:: 1.10.0

        .. note::
            This feature requires the SDL2 window provider.
        """
        pass

    def on_show(self, *largs):
        """Event called when the window is shown.

        .. versionadded:: 1.10.0

        .. note::
            This feature requires the SDL2 window provider.
        """
        pass

    def on_request_close(self, *largs, **kwargs):
        """Event called before we close the window. If a bound function returns
        `True`, the window will not be closed. If the event is triggered
        because of the keyboard escape key, the keyword argument `source` is
        dispatched along with a value of `keyboard` to the bound functions.

        .. warning::
            When the bound function returns True the window will not be closed,
            so use with care because the user would not be able to close the
            program, even if the red X is clicked.
        """
        pass

    def on_cursor_enter(self, *largs):
        """Event called when the cursor enters the window.

        .. versionadded:: 1.9.1

        .. note::
            This feature requires the SDL2 window provider.
        """
        pass

    def on_cursor_leave(self, *largs):
        """Event called when the cursor leaves the window.

        .. versionadded:: 1.9.1

        .. note::
            This feature requires the SDL2 window provider.
        """
        pass

    def on_mouse_down(self, x, y, button, modifiers):
        """Event called when the mouse is used (pressed/released)."""
        pass

    def on_mouse_move(self, x, y, modifiers):
        """Event called when the mouse is moved with buttons pressed."""
        pass

    def on_mouse_up(self, x, y, button, modifiers):
        """Event called when the mouse is moved with buttons pressed."""
        pass

    def on_joy_axis(self, stickid, axisid, value):
        """Event called when a joystick has a stick or other axis moved.

        .. versionadded:: 1.9.0"""
        pass

    def on_joy_hat(self, stickid, hatid, value):
        """Event called when a joystick has a hat/dpad moved.

        .. versionadded:: 1.9.0"""
        pass

    def on_joy_ball(self, stickid, ballid, xvalue, yvalue):
        """Event called when a joystick has a ball moved.

        .. versionadded:: 1.9.0"""
        pass

    def on_joy_button_down(self, stickid, buttonid):
        """Event called when a joystick has a button pressed.

        .. versionadded:: 1.9.0"""
        pass

    def on_joy_button_up(self, stickid, buttonid):
        """Event called when a joystick has a button released.

        .. versionadded:: 1.9.0"""
        pass

    def on_keyboard(self, key, scancode=None, codepoint=None, modifier=None, **kwargs):
        """Event called when keyboard is used.

        .. warning::
            Some providers may omit `scancode`, `codepoint` and/or `modifier`.
        """
        if 'unicode' in kwargs:
            Logger.warning('The use of the unicode parameter is deprecated, and will be removed in future versions. Use codepoint instead, which has identical semantics.')
        is_osx = platform == 'darwin'
        if key == 27 and platform == 'android':
            from android import mActivity
            mActivity.moveTaskToBack(True)
            return True
        elif WindowBase.on_keyboard.exit_on_escape:
            if key == 27 or all([is_osx, key in [113, 119], modifier == 1024]):
                if not self.dispatch('on_request_close', source='keyboard'):
                    stopTouchApp()
                    self.close()
                    return True
    if Config:
        on_keyboard.exit_on_escape = Config.getboolean('kivy', 'exit_on_escape')

        def __exit(section, name, value):
            WindowBase.__dict__['on_keyboard'].exit_on_escape = Config.getboolean('kivy', 'exit_on_escape')
        Config.add_callback(__exit, 'kivy', 'exit_on_escape')

    def on_key_down(self, key, scancode=None, codepoint=None, modifier=None, **kwargs):
        """Event called when a key is down (same arguments as on_keyboard)"""
        if 'unicode' in kwargs:
            Logger.warning('The use of the unicode parameter is deprecated, and will be removed in future versions. Use codepoint instead, which has identical semantics.')

    def on_key_up(self, key, scancode=None, codepoint=None, modifier=None, **kwargs):
        """Event called when a key is released (same arguments as on_keyboard).
        """
        if 'unicode' in kwargs:
            Logger.warning('The use of the unicode parameter is deprecated, and will be removed in future versions. Use codepoint instead, which has identical semantics.')

    def on_textinput(self, text):
        """Event called when text: i.e. alpha numeric non control keys or set
        of keys is entered. As it is not guaranteed whether we get one
        character or multiple ones, this event supports handling multiple
        characters.

        .. versionadded:: 1.9.0
        """
        pass

    def on_drop_begin(self, x, y, *args):
        """Event called when a text or a file drop on the application is about
        to begin. It will be followed-up by a single or a multiple
        `on_drop_text` or `on_drop_file` events ending with an `on_drop_end`
        event.

        Arguments `x` and `y` are the mouse cursor position at the time of the
        drop and you should only rely on them if the drop originated from the
        mouse.

        :Parameters:
            `x`: `int`
                Cursor x position, relative to the window :attr:`left`, at the
                time of the drop.
            `y`: `int`
                Cursor y position, relative to the window :attr:`top`, at the
                time of the drop.
            `*args`: `tuple`
                Additional arguments.

        .. note::
            This event works with sdl2 window provider.

        .. versionadded:: 2.1.0
        """
        pass

    def on_drop_file(self, filename, x, y, *args):
        """Event called when a file is dropped on the application.

        Arguments `x` and `y` are the mouse cursor position at the time of the
        drop and you should only rely on them if the drop originated from the
        mouse.

        :Parameters:
            `filename`: `bytes`
                Absolute path to a dropped file.
            `x`: `int`
                Cursor x position, relative to the window :attr:`left`, at the
                time of the drop.
            `y`: `int`
                Cursor y position, relative to the window :attr:`top`, at the
                time of the drop.
            `*args`: `tuple`
                Additional arguments.

        .. warning::
            This event currently works with sdl2 window provider, on pygame
            window provider and OS X with a patched version of pygame.
            This event is left in place for further evolution
            (ios, android etc.)

        .. note::
            On Windows it is possible to drop a file on the window title bar
            or on its edges and for that case :attr:`mouse_pos` won't be
            updated as the mouse cursor is not within the window.

        .. note::
            This event doesn't work for apps with elevated permissions,
            because the OS API calls are filtered. Check issue
            `#4999 <https://github.com/kivy/kivy/issues/4999>`_ for
            pointers to workarounds.

        .. versionadded:: 1.2.0

        .. versionchanged:: 2.1.0
            Renamed from `on_dropfile` to `on_drop_file`.
        """
        pass

    @deprecated(msg='Deprecated in 2.1.0, use on_drop_file event instead. Event on_dropfile will be removed in the next two releases.')
    def on_dropfile(self, filename):
        pass

    def on_drop_text(self, text, x, y, *args):
        """Event called when a text is dropped on the application.

        Arguments `x` and `y` are the mouse cursor position at the time of the
        drop and you should only rely on them if the drop originated from the
        mouse.

        :Parameters:
            `text`: `bytes`
                Text which is dropped.
            `x`: `int`
                Cursor x position, relative to the window :attr:`left`, at the
                time of the drop.
            `y`: `int`
                Cursor y position, relative to the window :attr:`top`, at the
                time of the drop.
            `*args`: `tuple`
                Additional arguments.

        .. note::
            This event works with sdl2 window provider on x11 window.

        .. note::
            On Windows it is possible to drop a text on the window title bar
            or on its edges and for that case :attr:`mouse_pos` won't be
            updated as the mouse cursor is not within the window.

        .. versionadded:: 2.1.0
        """
        pass

    def on_drop_end(self, x, y, *args):
        """Event called when a text or a file drop on the application has
        ended.

        Arguments `x` and `y` are the mouse cursor position at the time of the
        drop and you should only rely on them if the drop originated from the
        mouse.

        :Parameters:
            `x`: `int`
                Cursor x position, relative to the window :attr:`left`, at the
                time of the drop.
            `y`: `int`
                Cursor y position, relative to the window :attr:`top`, at the
                time of the drop.
            `*args`: `tuple`
                Additional arguments.

        .. note::
            This event works with sdl2 window provider.

        .. versionadded:: 2.1.0
        """
        pass

    def on_memorywarning(self):
        """Event called when the platform have memory issue.
        Your goal is to clear the cache in your app as much as you can,
        release unused widgets, do garbage collection etc.

        Currently, this event is fired only from the SDL2 provider, for
        iOS and Android.

        .. versionadded:: 1.9.0
        """
        pass

    def on_textedit(self, text):
        """Event called when inputting with IME.
        The string inputting with IME is set as the parameter of
        this event.

        .. versionadded:: 1.10.1
        """
        pass
    dpi = NumericProperty(96.0)
    "Return the DPI of the screen as computed by the window. If the\n    implementation doesn't support DPI lookup, it's 96.\n\n    .. warning::\n\n        This value is not cross-platform. Use\n        :attr:`kivy.metrics.Metrics.dpi` instead.\n    "

    def configure_keyboards(self):
        sk = self._system_keyboard
        self.bind(on_key_down=sk._on_window_key_down, on_key_up=sk._on_window_key_up, on_textinput=sk._on_window_textinput)
        self.use_syskeyboard = True
        self.allow_vkeyboard = False
        self.single_vkeyboard = True
        self.docked_vkeyboard = False
        mode = Config.get('kivy', 'keyboard_mode')
        if mode not in ('', 'system', 'dock', 'multi', 'systemanddock', 'systemandmulti'):
            Logger.critical('Window: unknown keyboard mode %r' % mode)
        if mode == 'system':
            self.use_syskeyboard = True
            self.allow_vkeyboard = False
            self.single_vkeyboard = True
            self.docked_vkeyboard = False
        elif mode == 'dock':
            self.use_syskeyboard = False
            self.allow_vkeyboard = True
            self.single_vkeyboard = True
            self.docked_vkeyboard = True
        elif mode == 'multi':
            self.use_syskeyboard = False
            self.allow_vkeyboard = True
            self.single_vkeyboard = False
            self.docked_vkeyboard = False
        elif mode == 'systemanddock':
            self.use_syskeyboard = True
            self.allow_vkeyboard = True
            self.single_vkeyboard = True
            self.docked_vkeyboard = True
        elif mode == 'systemandmulti':
            self.use_syskeyboard = True
            self.allow_vkeyboard = True
            self.single_vkeyboard = False
            self.docked_vkeyboard = False
        Logger.info('Window: virtual keyboard %sallowed, %s, %s' % ('' if self.allow_vkeyboard else 'not ', 'single mode' if self.single_vkeyboard else 'multiuser mode', 'docked' if self.docked_vkeyboard else 'not docked'))

    def set_vkeyboard_class(self, cls):
        """.. versionadded:: 1.0.8

        Set the VKeyboard class to use. If set to `None`, it will use the
        :class:`kivy.uix.vkeyboard.VKeyboard`.
        """
        self._vkeyboard_cls = cls

    def release_all_keyboards(self):
        """.. versionadded:: 1.0.8

        This will ensure that no virtual keyboard / system keyboard is
        requested. All instances will be closed.
        """
        for key in list(self._keyboards.keys())[:]:
            keyboard = self._keyboards[key]
            if keyboard:
                keyboard.release()

    def request_keyboard(self, callback, target, input_type='text', keyboard_suggestions=True):
        """.. versionadded:: 1.0.4

        Internal widget method to request the keyboard. This method is rarely
        required by the end-user as it is handled automatically by the
        :class:`~kivy.uix.textinput.TextInput`. We expose it in case you want
        to handle the keyboard manually for unique input scenarios.

        A widget can request the keyboard, indicating a callback to call
        when the keyboard is released (or taken by another widget).

        :Parameters:
            `callback`: func
                Callback that will be called when the keyboard is
                closed. This can be because somebody else requested the
                keyboard or the user closed it.
            `target`: Widget
                Attach the keyboard to the specified `target`. This should be
                the widget that requested the keyboard. Ensure you have a
                different target attached to each keyboard if you're working in
                a multi user mode.

                .. versionadded:: 1.0.8

            `input_type`: string
                Choose the type of soft keyboard to request. Can be one of
                'null', 'text', 'number', 'url', 'mail', 'datetime', 'tel',
                'address'.

                .. note::

                    `input_type` is currently only honored on Android.

                .. versionadded:: 1.8.0

                .. versionchanged:: 2.1.0
                    Added `null` to soft keyboard types.

            `keyboard_suggestions`: bool
                If True provides auto suggestions on top of keyboard.
                This will only work if input_type is set to `text`, `url`,
                `mail` or `address`.

                .. versionadded:: 2.1.0

        :Return:
            An instance of :class:`Keyboard` containing the callback, target,
            and if the configuration allows it, a
            :class:`~kivy.uix.vkeyboard.VKeyboard` instance attached as a
            *.widget* property.

        .. note::

            The behavior of this function is heavily influenced by the current
            `keyboard_mode`. Please see the Config's
            :ref:`configuration tokens <configuration-tokens>` section for
            more information.

        """
        self.release_keyboard(target)
        if self.allow_vkeyboard:
            keyboard = None
            global VKeyboard
            if VKeyboard is None and self._vkeyboard_cls is None:
                from kivy.uix.vkeyboard import VKeyboard
                self._vkeyboard_cls = VKeyboard
            key = 'single' if self.single_vkeyboard else target
            if key not in self._keyboards:
                vkeyboard = self._vkeyboard_cls()
                keyboard = Keyboard(widget=vkeyboard, window=self)
                vkeyboard.bind(on_key_down=keyboard._on_vkeyboard_key_down, on_key_up=keyboard._on_vkeyboard_key_up, on_textinput=keyboard._on_vkeyboard_textinput)
                self._keyboards[key] = keyboard
            else:
                keyboard = self._keyboards[key]
            keyboard.target = keyboard.widget.target = target
            keyboard.callback = keyboard.widget.callback = callback
            self.add_widget(keyboard.widget)
            keyboard.widget.docked = self.docked_vkeyboard
            keyboard.widget.setup_mode()
            if self.softinput_mode == 'pan':
                keyboard.widget.top = 0
            elif self.softinput_mode == 'below_target':
                keyboard.widget.top = keyboard.target.y
        else:
            keyboard = self._system_keyboard
            keyboard.callback = callback
            keyboard.target = target
        if self.allow_vkeyboard and self.use_syskeyboard:
            self.unbind(on_key_down=keyboard._on_window_key_down, on_key_up=keyboard._on_window_key_up, on_textinput=keyboard._on_window_textinput)
            self.bind(on_key_down=keyboard._on_window_key_down, on_key_up=keyboard._on_window_key_up, on_textinput=keyboard._on_window_textinput)
        return keyboard

    def release_keyboard(self, target=None):
        """.. versionadded:: 1.0.4

        Internal method for the widget to release the real-keyboard. Check
        :meth:`request_keyboard` to understand how it works.
        """
        if self.allow_vkeyboard:
            key = 'single' if self.single_vkeyboard else target
            if key not in self._keyboards:
                return
            keyboard = self._keyboards[key]
            callback = keyboard.callback
            if callback:
                keyboard.callback = None
                callback()
            keyboard.target = None
            self.remove_widget(keyboard.widget)
            if key != 'single' and key in self._keyboards:
                del self._keyboards[key]
        elif self._system_keyboard.callback:
            callback = self._system_keyboard.callback
            self._system_keyboard.callback = None
            callback()
            return True

    def grab_mouse(self):
        """Grab mouse - so won't leave window

        .. versionadded:: 1.10.0

        .. note::
            This feature requires the SDL2 window provider.
        """
        pass

    def ungrab_mouse(self):
        """Ungrab mouse

        .. versionadded:: 1.10.0

        .. note::
            This feature requires the SDL2 window provider.
        """
        pass