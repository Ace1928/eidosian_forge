from os.path import join
import sys
from typing import Optional
from kivy import kivy_data_dir
from kivy.logger import Logger
from kivy.base import EventLoop
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import WindowBase
from kivy.input.provider import MotionEventProvider
from kivy.input.motionevent import MotionEvent
from kivy.resources import resource_find
from kivy.utils import platform, deprecated
from kivy.compat import unichr
from collections import deque
class WindowSDL(WindowBase):
    _win_dpi_watch: Optional['_WindowsSysDPIWatch'] = None
    _do_resize_ev = None
    managed_textinput = True

    def __init__(self, **kwargs):
        self._pause_loop = False
        self._cursor_entered = False
        self._drop_pos = None
        self._win = _WindowSDL2Storage()
        super(WindowSDL, self).__init__()
        self.titlebar_widget = None
        self._mouse_x = self._mouse_y = -1
        self._meta_keys = (KMOD_LCTRL, KMOD_RCTRL, KMOD_RSHIFT, KMOD_LSHIFT, KMOD_RALT, KMOD_LALT, KMOD_LGUI, KMOD_RGUI, KMOD_NUM, KMOD_CAPS, KMOD_MODE)
        self.command_keys = {27: 'escape', 9: 'tab', 8: 'backspace', 13: 'enter', 127: 'del', 271: 'enter', 273: 'up', 274: 'down', 275: 'right', 276: 'left', 278: 'home', 279: 'end', 280: 'pgup', 281: 'pgdown'}
        self._mouse_buttons_down = set()
        self.key_map = {SDLK_LEFT: 276, SDLK_RIGHT: 275, SDLK_UP: 273, SDLK_DOWN: 274, SDLK_HOME: 278, SDLK_END: 279, SDLK_PAGEDOWN: 281, SDLK_PAGEUP: 280, SDLK_SHIFTR: 303, SDLK_SHIFTL: 304, SDLK_SUPER: 309, SDLK_LCTRL: 305, SDLK_RCTRL: 306, SDLK_LALT: 308, SDLK_RALT: 307, SDLK_CAPS: 301, SDLK_INSERT: 277, SDLK_F1: 282, SDLK_F2: 283, SDLK_F3: 284, SDLK_F4: 285, SDLK_F5: 286, SDLK_F6: 287, SDLK_F7: 288, SDLK_F8: 289, SDLK_F9: 290, SDLK_F10: 291, SDLK_F11: 292, SDLK_F12: 293, SDLK_F13: 294, SDLK_F14: 295, SDLK_F15: 296, SDLK_KEYPADNUM: 300, SDLK_KP_DIVIDE: 267, SDLK_KP_MULTIPLY: 268, SDLK_KP_MINUS: 269, SDLK_KP_PLUS: 270, SDLK_KP_ENTER: 271, SDLK_KP_DOT: 266, SDLK_KP_0: 256, SDLK_KP_1: 257, SDLK_KP_2: 258, SDLK_KP_3: 259, SDLK_KP_4: 260, SDLK_KP_5: 261, SDLK_KP_6: 262, SDLK_KP_7: 263, SDLK_KP_8: 264, SDLK_KP_9: 265}
        if platform == 'ios':
            self.key_map[127] = 8
        elif platform == 'android':
            self.key_map[1073742094] = 27
        self.bind(minimum_width=self._set_minimum_size, minimum_height=self._set_minimum_size)
        self.bind(allow_screensaver=self._set_allow_screensaver)
        self.bind(always_on_top=self._set_always_on_top)

    def get_window_info(self):
        return self._win.get_window_info()

    def _set_minimum_size(self, *args):
        minimum_width = self.minimum_width
        minimum_height = self.minimum_height
        if minimum_width and minimum_height:
            self._win.set_minimum_size(minimum_width, minimum_height)
        elif minimum_width or minimum_height:
            Logger.warning('Both Window.minimum_width and Window.minimum_height must be bigger than 0 for the size restriction to take effect.')

    def _set_always_on_top(self, *args):
        self._win.set_always_on_top(self.always_on_top)

    def _set_allow_screensaver(self, *args):
        self._win.set_allow_screensaver(self.allow_screensaver)

    def _event_filter(self, action, *largs):
        from kivy.app import App
        if action == 'app_terminating':
            EventLoop.quit = True
        elif action == 'app_lowmemory':
            self.dispatch('on_memorywarning')
        elif action == 'app_willenterbackground':
            from kivy.base import stopTouchApp
            app = App.get_running_app()
            if not app:
                Logger.info('WindowSDL: No running App found, pause.')
            elif not app.dispatch('on_pause'):
                if platform == 'android':
                    Logger.info('WindowSDL: App stopped, on_pause() returned False.')
                    from android import mActivity
                    mActivity.finishAndRemoveTask()
                else:
                    Logger.info("WindowSDL: App doesn't support pause mode, stop.")
                    stopTouchApp()
                    return 0
            self._pause_loop = True
        elif action == 'app_didenterforeground':
            if self._pause_loop:
                self._pause_loop = False
                app = App.get_running_app()
                if app:
                    app.dispatch('on_resume')
        elif action == 'windowresized':
            self._size = largs
            self._win.resize_window(*self._size)
            EventLoop.idle()
        return 0

    def create_window(self, *largs):
        if self._fake_fullscreen:
            if not self.borderless:
                self.fullscreen = self._fake_fullscreen = False
            elif not self.fullscreen or self.fullscreen == 'auto':
                self.custom_titlebar = self.borderless = self._fake_fullscreen = False
            elif self.custom_titlebar:
                if platform == 'win':
                    self.borderless = False
        if self.fullscreen == 'fake':
            self.borderless = self._fake_fullscreen = True
            Logger.warning("The 'fake' fullscreen option has been deprecated, use Window.borderless or the borderless Config option instead.")
        if not self.initialized:
            if self.position == 'auto':
                pos = (None, None)
            elif self.position == 'custom':
                pos = (self.left, self.top)
            self._win.set_event_filter(self._event_filter)
            w, h = self.system_size
            resizable = Config.getboolean('graphics', 'resizable')
            state = Config.get('graphics', 'window_state') if self._is_desktop else None
            self.system_size = self._win.setup_window(pos[0], pos[1], w, h, self.borderless, self.fullscreen, resizable, state, self.get_gl_backend_name())
            self._update_density_and_dpi()
            self._pos = (0, 0)
            self._set_minimum_size()
            self._set_allow_screensaver()
            self._set_always_on_top()
            if state == 'hidden':
                self._focus = False
        else:
            w, h = self.system_size
            self._win.resize_window(w, h)
            if platform == 'win':
                if self.custom_titlebar:
                    if Config.getboolean('graphics', 'resizable'):
                        import win32con
                        import ctypes
                        self._win.set_border_state(False)
                        ctypes.windll.user32.SetWindowPos(self._win.get_window_info().window, win32con.HWND_TOP, *self._win.get_window_pos(), *self.system_size, win32con.SWP_FRAMECHANGED)
                    else:
                        self._win.set_border_state(True)
                else:
                    self._win.set_border_state(self.borderless)
            else:
                self._win.set_border_state(self.borderless or self.custom_titlebar)
            self._win.set_fullscreen_mode(self.fullscreen)
        super(WindowSDL, self).create_window()
        self._set_cursor_state(self.show_cursor)
        if self.initialized:
            return
        Logger.info('Window: auto add sdl2 input provider')
        SDL2MotionEventProvider.win = self
        EventLoop.add_input_provider(SDL2MotionEventProvider('sdl', ''))
        try:
            filename_icon = self.icon or Config.get('kivy', 'window_icon')
            if filename_icon == '':
                logo_size = 32
                if platform == 'macosx':
                    logo_size = 512
                elif platform == 'win':
                    logo_size = 64
                filename_icon = 'kivy-icon-{}.png'.format(logo_size)
                filename_icon = resource_find(join(kivy_data_dir, 'logo', filename_icon))
            self.set_icon(filename_icon)
        except:
            Logger.exception('Window: cannot set icon')
        if platform == 'win' and self._win_dpi_watch is None:
            self._win_dpi_watch = _WindowsSysDPIWatch(window=self)
            self._win_dpi_watch.start()

    def _update_density_and_dpi(self):
        if platform == 'win':
            from ctypes import windll
            self._density = 1.0
            try:
                hwnd = windll.user32.GetActiveWindow()
                self.dpi = float(windll.user32.GetDpiForWindow(hwnd))
                self._density = self.dpi / 96
            except AttributeError:
                pass
        else:
            self._density = self._win._get_gl_size()[0] / self._size[0]
            if self._is_desktop:
                self.dpi = self._density * 96.0

    def close(self):
        self._win.teardown_window()
        super(WindowSDL, self).close()
        if self._win_dpi_watch is not None:
            self._win_dpi_watch.stop()
            self._win_dpi_watch = None
        self.initialized = False

    def maximize(self):
        if self._is_desktop:
            self._win.maximize_window()
        else:
            Logger.warning('Window: maximize() is used only on desktop OSes.')

    def minimize(self):
        if self._is_desktop:
            self._win.minimize_window()
        else:
            Logger.warning('Window: minimize() is used only on desktop OSes.')

    def restore(self):
        if self._is_desktop:
            self._win.restore_window()
        else:
            Logger.warning('Window: restore() is used only on desktop OSes.')

    def hide(self):
        if self._is_desktop:
            self._win.hide_window()
        else:
            Logger.warning('Window: hide() is used only on desktop OSes.')

    def show(self):
        if self._is_desktop:
            self._win.show_window()
        else:
            Logger.warning('Window: show() is used only on desktop OSes.')

    def raise_window(self):
        if self._is_desktop:
            self._win.raise_window()
        else:
            Logger.warning('Window: show() is used only on desktop OSes.')

    def set_title(self, title):
        self._win.set_window_title(title)

    def set_icon(self, filename):
        self._win.set_window_icon(str(filename))

    def screenshot(self, *largs, **kwargs):
        filename = super(WindowSDL, self).screenshot(*largs, **kwargs)
        if filename is None:
            return
        from kivy.graphics.opengl import glReadPixels, GL_RGB, GL_UNSIGNED_BYTE
        width, height = self.size
        data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        self._win.save_bytes_in_png(filename, data, width, height)
        Logger.debug('Window: Screenshot saved at <%s>' % filename)
        return filename

    def flip(self):
        self._win.flip()
        super(WindowSDL, self).flip()

    def set_system_cursor(self, cursor_name):
        result = self._win.set_system_cursor(cursor_name)
        return result

    def _get_window_pos(self):
        return self._win.get_window_pos()

    def _set_window_pos(self, x, y):
        self._win.set_window_pos(x, y)

    def _get_window_opacity(self):
        return self._win.get_window_opacity()

    def _set_window_opacity(self, opacity):
        if self.opacity != opacity:
            return self._win.set_window_opacity(opacity)

    def _is_shaped(self):
        return self._win.is_window_shaped()

    def _set_shape(self, shape_image, mode='default', cutoff=False, color_key=None):
        modes = ('default', 'binalpha', 'reversebinalpha', 'colorkey')
        color_key = color_key or (0, 0, 0, 1)
        if mode not in modes:
            Logger.warning('Window: shape mode can be only {}'.format(', '.join(modes)))
            return
        if not isinstance(color_key, (tuple, list)):
            return
        if len(color_key) not in (3, 4):
            return
        if len(color_key) == 3:
            color_key = (color_key[0], color_key[1], color_key[2], 1)
            Logger.warning('Window: Shape color_key must be only tuple or list')
            return
        color_key = (color_key[0] * 255, color_key[1] * 255, color_key[2] * 255, color_key[3] * 255)
        assert cutoff in (1, 0)
        shape_image = shape_image or Config.get('kivy', 'window_shape')
        shape_image = resource_find(shape_image) or shape_image
        self._win.set_shape(shape_image, mode, cutoff, color_key)

    def _get_shaped_mode(self):
        return self._win.get_shaped_mode()

    def _set_shaped_mode(self, value):
        self._set_shape(shape_image=self.shape_image, mode=value, cutoff=self.shape_cutoff, color_key=self.shape_color_key)
        return self._win.get_shaped_mode()

    def _set_cursor_state(self, value):
        self._win._set_cursor_state(value)

    def _fix_mouse_pos(self, x, y):
        self.mouse_pos = (x * self._density, (self.system_size[1] - 1 - y) * self._density)
        return (x, y)

    def mainloop(self):
        while self._pause_loop:
            self._win.wait_event()
            if not self._pause_loop:
                break
            event = self._win.poll()
            if event is None:
                continue
            action, args = (event[0], event[1:])
            if action.startswith('drop'):
                self._dispatch_drop_event(action, args)
            elif EventLoop.quit:
                return
        while True:
            event = self._win.poll()
            if event is False:
                break
            if event is None:
                continue
            action, args = (event[0], event[1:])
            if action == 'quit':
                if self.dispatch('on_request_close'):
                    continue
                EventLoop.quit = True
                break
            elif action in ('fingermotion', 'fingerdown', 'fingerup'):
                if platform in ('ios', 'android'):
                    SDL2MotionEventProvider.q.appendleft(event)
                pass
            elif action == 'mousemotion':
                x, y = args
                x, y = self._fix_mouse_pos(x, y)
                self._mouse_x = x
                self._mouse_y = y
                if not self._cursor_entered:
                    self._cursor_entered = True
                    self.dispatch('on_cursor_enter')
                if len(self._mouse_buttons_down) == 0:
                    continue
                self._mouse_meta = self.modifiers
                self.dispatch('on_mouse_move', x, y, self.modifiers)
            elif action in ('mousebuttondown', 'mousebuttonup'):
                x, y, button = args
                x, y = self._fix_mouse_pos(x, y)
                self._mouse_x = x
                self._mouse_y = y
                if not self._cursor_entered:
                    self._cursor_entered = True
                    self.dispatch('on_cursor_enter')
                btn = 'left'
                if button == 3:
                    btn = 'right'
                elif button == 2:
                    btn = 'middle'
                elif button == 4:
                    btn = 'mouse4'
                elif button == 5:
                    btn = 'mouse5'
                eventname = 'on_mouse_down'
                self._mouse_buttons_down.add(button)
                if action == 'mousebuttonup':
                    eventname = 'on_mouse_up'
                    self._mouse_buttons_down.remove(button)
                self.dispatch(eventname, x, y, btn, self.modifiers)
            elif action.startswith('mousewheel'):
                x, y = self._win.get_relative_mouse_pos()
                if not self._collide_and_dispatch_cursor_enter(x, y):
                    continue
                self._update_modifiers()
                x, y, button = args
                btn = 'scrolldown'
                if action.endswith('up'):
                    btn = 'scrollup'
                elif action.endswith('right'):
                    btn = 'scrollright'
                elif action.endswith('left'):
                    btn = 'scrollleft'
                self._mouse_meta = self.modifiers
                self._mouse_btn = btn
                self._mouse_down = True
                self.dispatch('on_mouse_down', self._mouse_x, self._mouse_y, btn, self.modifiers)
                self._mouse_down = False
                self.dispatch('on_mouse_up', self._mouse_x, self._mouse_y, btn, self.modifiers)
            elif action.startswith('drop'):
                self._dispatch_drop_event(action, args)
            elif action == 'windowresized':
                self._size = self._win.window_size
                ev = self._do_resize_ev
                if ev is None:
                    ev = Clock.schedule_once(self._do_resize, 0.1)
                    self._do_resize_ev = ev
                else:
                    ev()
            elif action == 'windowdisplaychanged':
                Logger.info(f'WindowSDL: Window is now on display {args[0]}')
                self._update_density_and_dpi()
            elif action == 'windowmoved':
                self.dispatch('on_move')
            elif action == 'windowrestored':
                self.dispatch('on_restore')
                self.canvas.ask_update()
            elif action == 'windowexposed':
                self.canvas.ask_update()
            elif action == 'windowminimized':
                self.dispatch('on_minimize')
                if Config.getboolean('kivy', 'pause_on_minimize'):
                    self.do_pause()
            elif action == 'windowmaximized':
                self.dispatch('on_maximize')
            elif action == 'windowhidden':
                self.dispatch('on_hide')
            elif action == 'windowshown':
                self.dispatch('on_show')
            elif action == 'windowfocusgained':
                self._focus = True
            elif action == 'windowfocuslost':
                self._focus = False
            elif action == 'windowenter':
                x, y = self._win.get_relative_mouse_pos()
                self._collide_and_dispatch_cursor_enter(x, y)
            elif action == 'windowleave':
                self._cursor_entered = False
                self.dispatch('on_cursor_leave')
            elif action == 'joyaxismotion':
                stickid, axisid, value = args
                self.dispatch('on_joy_axis', stickid, axisid, value)
            elif action == 'joyhatmotion':
                stickid, hatid, value = args
                self.dispatch('on_joy_hat', stickid, hatid, value)
            elif action == 'joyballmotion':
                stickid, ballid, xrel, yrel = args
                self.dispatch('on_joy_ball', stickid, ballid, xrel, yrel)
            elif action == 'joybuttondown':
                stickid, buttonid = args
                self.dispatch('on_joy_button_down', stickid, buttonid)
            elif action == 'joybuttonup':
                stickid, buttonid = args
                self.dispatch('on_joy_button_up', stickid, buttonid)
            elif action in ('keydown', 'keyup'):
                mod, key, scancode, kstr = args
                try:
                    key = self.key_map[key]
                except KeyError:
                    pass
                if action == 'keydown':
                    self._update_modifiers(mod, key)
                else:
                    self._update_modifiers(mod)
                if key not in self._modifiers and key not in self.command_keys.keys():
                    try:
                        kstr_chr = unichr(key)
                        try:
                            encoding = getattr(sys.stdout, 'encoding', 'utf8') or 'utf8'
                            kstr_chr.encode(encoding)
                            kstr = kstr_chr
                        except UnicodeError:
                            pass
                    except ValueError:
                        pass
                if action == 'keyup':
                    self.dispatch('on_key_up', key, scancode)
                    continue
                if self.dispatch('on_key_down', key, scancode, kstr, self.modifiers):
                    continue
                self.dispatch('on_keyboard', key, scancode, kstr, self.modifiers)
            elif action == 'textinput':
                text = args[0]
                self.dispatch('on_textinput', text)
            elif action == 'textedit':
                text = args[0]
                self.dispatch('on_textedit', text)
            else:
                Logger.trace('WindowSDL: Unhandled event %s' % str(event))

    def _dispatch_drop_event(self, action, args):
        x, y = (0, 0) if self._drop_pos is None else self._drop_pos
        if action == 'dropfile':
            self.dispatch('on_drop_file', args[0], x, y)
        elif action == 'droptext':
            self.dispatch('on_drop_text', args[0], x, y)
        elif action == 'dropbegin':
            self._drop_pos = x, y = self._win.get_relative_mouse_pos()
            self._collide_and_dispatch_cursor_enter(x, y)
            self.dispatch('on_drop_begin', x, y)
        elif action == 'dropend':
            self._drop_pos = None
            self.dispatch('on_drop_end', x, y)

    def _collide_and_dispatch_cursor_enter(self, x, y):
        w, h = self._win.window_size
        if 0 <= x < w and 0 <= y < h:
            self._mouse_x, self._mouse_y = self._fix_mouse_pos(x, y)
            if not self._cursor_entered:
                self._cursor_entered = True
                self.dispatch('on_cursor_enter')
            return True

    def _do_resize(self, dt):
        Logger.debug('Window: Resize window to %s' % str(self.size))
        self._win.resize_window(*self._size)
        self.dispatch('on_pre_resize', *self.size)

    def do_pause(self):
        from kivy.app import App
        from kivy.base import stopTouchApp
        app = App.get_running_app()
        if not app:
            Logger.info('WindowSDL: No running App found, pause.')
        elif not app.dispatch('on_pause'):
            Logger.info("WindowSDL: App doesn't support pause mode, stop.")
            stopTouchApp()
            return
        while True:
            event = self._win.poll()
            if event is False:
                continue
            if event is None:
                continue
            action, args = (event[0], event[1:])
            if action == 'quit':
                EventLoop.quit = True
                break
            elif action == 'app_willenterforeground':
                break
            elif action == 'windowrestored':
                break
        if app:
            app.dispatch('on_resume')

    def _update_modifiers(self, mods=None, key=None):
        if mods is None and key is None:
            return
        modifiers = set()
        if mods is not None:
            if mods & (KMOD_RSHIFT | KMOD_LSHIFT):
                modifiers.add('shift')
            if mods & (KMOD_RALT | KMOD_LALT | KMOD_MODE):
                modifiers.add('alt')
            if mods & (KMOD_RCTRL | KMOD_LCTRL):
                modifiers.add('ctrl')
            if mods & (KMOD_RGUI | KMOD_LGUI):
                modifiers.add('meta')
            if mods & KMOD_NUM:
                modifiers.add('numlock')
            if mods & KMOD_CAPS:
                modifiers.add('capslock')
        if key is not None:
            if key in (KMOD_RSHIFT, KMOD_LSHIFT):
                modifiers.add('shift')
            if key in (KMOD_RALT, KMOD_LALT, KMOD_MODE):
                modifiers.add('alt')
            if key in (KMOD_RCTRL, KMOD_LCTRL):
                modifiers.add('ctrl')
            if key in (KMOD_RGUI, KMOD_LGUI):
                modifiers.add('meta')
            if key == KMOD_NUM:
                modifiers.add('numlock')
            if key == KMOD_CAPS:
                modifiers.add('capslock')
        self._modifiers = list(modifiers)
        return

    def request_keyboard(self, callback, target, input_type='text', keyboard_suggestions=True):
        self._sdl_keyboard = super(WindowSDL, self).request_keyboard(callback, target, input_type, keyboard_suggestions)
        self._win.show_keyboard(self._system_keyboard, self.softinput_mode, input_type, keyboard_suggestions)
        Clock.schedule_interval(self._check_keyboard_shown, 1 / 5.0)
        return self._sdl_keyboard

    def release_keyboard(self, *largs):
        super(WindowSDL, self).release_keyboard(*largs)
        self._win.hide_keyboard()
        self._sdl_keyboard = None
        return True

    def _check_keyboard_shown(self, dt):
        if self._sdl_keyboard is None:
            return False
        if not self._win.is_keyboard_shown():
            self._sdl_keyboard.release()

    def map_key(self, original_key, new_key):
        self.key_map[original_key] = new_key

    def unmap_key(self, key):
        if key in self.key_map:
            del self.key_map[key]

    def grab_mouse(self):
        self._win.grab_mouse(True)

    def ungrab_mouse(self):
        self._win.grab_mouse(False)

    def set_custom_titlebar(self, titlebar_widget):
        if not self.custom_titlebar:
            Logger.warning("Window: Window.custom_titlebar not set to True… can't set custom titlebar")
            return
        self.titlebar_widget = titlebar_widget
        return self._win.set_custom_titlebar(self.titlebar_widget) == 0