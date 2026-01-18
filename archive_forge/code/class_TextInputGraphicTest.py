import unittest
from itertools import count
from kivy.core.window import Window
from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.uix.textinput import TextInput, TextInputCutCopyPaste
from kivy.uix.widget import Widget
from kivy.clock import Clock
class TextInputGraphicTest(GraphicUnitTest):

    def test_text_validate(self):
        ti = TextInput(multiline=False)
        ti.focus = True
        self.render(ti)
        self.assertFalse(ti.multiline)
        self.assertTrue(ti.focus)
        self.assertTrue(ti.text_validate_unfocus)
        ti.validate_test = None
        ti.bind(on_text_validate=lambda *_: setattr(ti, 'validate_test', True))
        ti._key_down((None, None, 'enter', 1), repeat=False)
        self.assertTrue(ti.validate_test)
        self.assertFalse(ti.focus)
        ti.validate_test = None
        ti.text_validate_unfocus = False
        ti.focus = True
        self.assertTrue(ti.focus)
        ti._key_down((None, None, 'enter', 1), repeat=False)
        self.assertTrue(ti.validate_test)
        self.assertTrue(ti.focus)

    def test_selection_enter_multiline(self):
        text = 'multiline\ntext'
        ti = TextInput(multiline=True, text=text)
        ti.focus = True
        self.render(ti)
        self.assertTrue(ti.focus)
        self.assertEqual(ti.cursor, (len(text.split('\n')[-1]), len(text.split('\n')) - 1))
        ti._key_down((None, None, 'shift', 1), repeat=False)
        ti._key_down((None, None, 'cursor_up', 1), repeat=False)
        ti._key_up((None, None, 'shift', 1), repeat=False)
        self.assertEqual(ti.cursor, (len(text.split('\n')[-1]), len(text.split('\n')) - 2))
        self.assertEqual(ti.text, text)
        ti._key_down((None, None, 'enter', 1), repeat=False)
        self.assertEqual(ti.text, text[:4] + '\n')

    def test_selection_enter_singleline(self):
        text = 'singleline'
        ti = TextInput(multiline=False, text=text)
        ti.focus = True
        self.render(ti)
        self.assertTrue(ti.focus)
        self.assertEqual(ti.cursor, (len(text), 0))
        steps = 4
        options = (('enter', text), ('backspace', text[:len(text) - steps]))
        for key, txt in options:
            ti._key_down((None, None, 'shift', 1), repeat=False)
            for _ in range(steps):
                ti._key_down((None, None, 'cursor_left', 1), repeat=False)
            ti._key_up((None, None, 'shift', 1), repeat=False)
            self.assertEqual(ti.cursor, (len(text[:-steps]), 0))
            self.assertEqual(ti.text, text)
            ti._key_down((None, None, key, 1), repeat=False)
            self.assertEqual(ti.text, txt)
            ti._key_down((None, None, 'cursor_end', 1), repeat=False)

    def test_del(self):
        text = 'some_random_text'
        ti = TextInput(multiline=False, text=text)
        ti.focus = True
        self.render(ti)
        self.assertTrue(ti.focus)
        self.assertEqual(ti.cursor, (len(text), 0))
        steps_skip = 2
        steps_select = 4
        del_key = 'del'
        for _ in range(steps_skip):
            ti._key_down((None, None, 'cursor_left', 1), repeat=False)
        ti._key_down((None, None, 'shift', 1), repeat=False)
        for _ in range(steps_select):
            ti._key_down((None, None, 'cursor_left', 1), repeat=False)
        ti._key_up((None, None, 'shift', 1), repeat=False)
        self.assertEqual(ti.cursor, (len(text[:-steps_select - steps_skip]), 0))
        self.assertEqual(ti.text, text)
        ti._key_down((None, None, del_key, 1), repeat=False)
        self.assertEqual(ti.text, 'some_randoxt')
        ti._key_down((None, None, del_key, 1), repeat=False)
        self.assertEqual(ti.text, 'some_randot')

    def test_escape(self):
        text = 'some_random_text'
        escape_key = 'escape'
        ti = TextInput(multiline=False, text=text)
        ti.focus = True
        self.render(ti)
        self.assertTrue(ti.focus)
        ti._key_down((None, None, escape_key, 1), repeat=False)
        self.assertFalse(ti.focus)
        self.assertEqual(ti.text, text)

    def test_no_shift_cursor_arrow_on_selection(self):
        text = 'some_random_text'
        ti = TextInput(multiline=False, text=text)
        ti.focus = True
        self.render(ti)
        self.assertTrue(ti.focus)
        self.assertEqual(ti.cursor, (len(text), 0))
        steps_skip = 2
        steps_select = 4
        for _ in range(steps_skip):
            ti._key_down((None, None, 'cursor_left', 1), repeat=False)
        ti._key_down((None, None, 'shift', 1), repeat=False)
        for _ in range(steps_select):
            ti._key_down((None, None, 'cursor_left', 1), repeat=False)
        ti._key_up((None, None, 'shift', 1), repeat=False)
        ti._key_down((None, None, 'cursor_right', 1), repeat=False)
        self.assertEqual(ti.cursor, (len(text) - steps_skip, 0))

    def test_cursor_movement_control(self):
        text = 'these are\nmany words'
        ti = TextInput(multiline=True, text=text)
        ti.focus = True
        self.render(ti)
        self.assertTrue(ti.focus)
        self.assertEqual(ti.cursor, (len(text.split('\n')[-1]), len(text.split('\n')) - 1))
        options = (('cursor_left', (5, 1)), ('cursor_left', (0, 1)), ('cursor_left', (6, 0)), ('cursor_right', (9, 0)), ('cursor_right', (4, 1)))
        for key, pos in options:
            ti._key_down((None, None, 'ctrl_L', 1), repeat=False)
            ti._key_down((None, None, key, 1), repeat=False)
            self.assertEqual(ti.cursor, pos)
            ti._key_up((None, None, 'ctrl_L', 1), repeat=False)

    def test_cursor_blink(self):
        ti = TextInput(cursor_blink=True)
        ti.focus = True
        ti._do_blink_cursor_ev = Clock.create_trigger(ti._do_blink_cursor, 0.01, interval=True)
        self.render(ti)
        self.assertTrue(ti.cursor_blink)
        self.assertTrue(ti._do_blink_cursor_ev.is_triggered)
        ti.cursor_blink = False
        for i in range(30):
            self.advance_frames(int(0.01 * Clock._max_fps) + 1)
            self.assertFalse(ti._do_blink_cursor_ev.is_triggered)
            self.assertFalse(ti._cursor_blink)
        ti.cursor_blink = True
        self.assertTrue(ti.cursor_blink)
        for i in range(30):
            self.advance_frames(int(0.01 * Clock._max_fps) + 1)
            self.assertTrue(ti._do_blink_cursor_ev.is_triggered)

    def test_visible_lines_range(self):
        ti = self.make_scrollable_text_input()
        assert ti._visible_lines_range == (20, 30)
        ti.height = ti_height_for_x_lines(ti, 2.5)
        ti.do_cursor_movement('cursor_home', control=True)
        self.advance_frames(1)
        assert ti._visible_lines_range == (0, 3)
        ti.height = ti_height_for_x_lines(ti, 0)
        self.advance_frames(1)
        assert ti._visible_lines_range == (0, 0)

    def test_keyboard_scroll(self):
        ti = self.make_scrollable_text_input()
        prev_cursor = ti.cursor
        ti.do_cursor_movement('cursor_home', control=True)
        self.advance_frames(1)
        assert ti._visible_lines_range == (0, 10)
        assert prev_cursor != ti.cursor
        prev_cursor = ti.cursor
        ti.do_cursor_movement('cursor_down', control=True)
        self.advance_frames(1)
        assert ti._visible_lines_range == (1, 11)
        assert prev_cursor == ti.cursor
        prev_cursor = ti.cursor
        ti.do_cursor_movement('cursor_up', control=True)
        self.advance_frames(1)
        assert ti._visible_lines_range == (0, 10)
        assert prev_cursor == ti.cursor
        prev_cursor = ti.cursor
        ti.do_cursor_movement('cursor_end', control=True)
        self.advance_frames(1)
        assert ti._visible_lines_range == (20, 30)
        assert prev_cursor != ti.cursor

    def test_scroll_doesnt_move_cursor(self):
        ti = self.make_scrollable_text_input()
        from kivy.base import EventLoop
        win = EventLoop.window
        touch = UTMotionEvent('unittest', next(touch_id), {'x': ti.center_x / float(win.width), 'y': ti.center_y / float(win.height)})
        touch.profile.append('button')
        touch.button = 'scrolldown'
        prev_cursor = ti.cursor
        assert ti._visible_lines_range == (20, 30)
        EventLoop.post_dispatch_input('begin', touch)
        EventLoop.post_dispatch_input('end', touch)
        self.advance_frames(1)
        assert ti._visible_lines_range == (20 - ti.lines_to_scroll, 30 - ti.lines_to_scroll)
        assert ti.cursor == prev_cursor

    def test_vertical_scroll_doesnt_depend_on_lines_rendering(self):
        ti = self.make_scrollable_text_input()
        ti.do_cursor_movement('cursor_home', control=True)
        self.advance_frames(1)
        assert ti._visible_lines_range == (0, 10)
        from kivy.base import EventLoop
        win = EventLoop.window
        for _ in range(0, 30, ti.lines_to_scroll):
            touch = UTMotionEvent('unittest', next(touch_id), {'x': ti.center_x / float(win.width), 'y': ti.center_y / float(win.height)})
            touch.profile.append('button')
            touch.button = 'scrollup'
            EventLoop.post_dispatch_input('begin', touch)
            EventLoop.post_dispatch_input('end', touch)
            self.advance_frames(1)
        assert ti._visible_lines_range == (20, 30)
        ti.do_cursor_movement('cursor_home', control=True)
        ti._trigger_update_graphics()
        self.advance_frames(1)
        assert ti._visible_lines_range == (0, 10)
        touch = UTMotionEvent('unittest', next(touch_id), {'x': ti.center_x / float(win.width), 'y': ti.center_y / float(win.height)})
        touch.profile.append('button')
        touch.button = 'scrollup'
        EventLoop.post_dispatch_input('begin', touch)
        EventLoop.post_dispatch_input('end', touch)
        self.advance_frames(1)
        assert ti._visible_lines_range == (ti.lines_to_scroll, 10 + ti.lines_to_scroll)

    def test_selectall_copy_paste(self):
        text = 'test'
        ti = TextInput(multiline=False, text=text)
        ti.focus = True
        self.render(ti)
        from kivy.base import EventLoop
        win = EventLoop.window
        win.dispatch('on_key_down', 97, 4, 'a', ['capslock', 'ctrl'])
        win.dispatch('on_key_up', 97, 4)
        self.advance_frames(1)
        win.dispatch('on_key_down', 99, 6, 'c', ['capslock', 'numlock', 'ctrl'])
        win.dispatch('on_key_up', 99, 6)
        self.advance_frames(1)
        win.dispatch('on_key_down', 278, 74, None, ['capslock'])
        win.dispatch('on_key_up', 278, 74)
        self.advance_frames(1)
        win.dispatch('on_key_down', 118, 25, 'v', ['numlock', 'ctrl'])
        win.dispatch('on_key_up', 118, 25)
        self.advance_frames(1)
        assert ti.text == 'testtest'

    def make_scrollable_text_input(self, num_of_lines=30, n_lines_to_show=10):
        """Prepare and start rendering the scrollable text input.

           num_of_lines -- amount of dummy lines used as contents
           n_lines_to_show -- amount of lines to fit in viewport
        """
        text = '\n'.join(map(str, range(num_of_lines)))
        ti = TextInput(text=text)
        ti.focus = True
        container = Widget()
        container.add_widget(ti)
        self.render(container)
        ti.height = ti_height_for_x_lines(ti, n_lines_to_show)
        self.advance_frames(1)
        return ti

    def test_cutcopypastebubble_content(self):
        tibubble = TextInputCutCopyPaste(textinput=TextInput())
        assert tibubble.but_copy.parent == tibubble.content
        assert tibubble.but_cut.parent == tibubble.content
        assert tibubble.but_paste.parent == tibubble.content
        assert tibubble.but_selectall.parent == tibubble.content