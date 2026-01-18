import unittest
from itertools import count
from kivy.core.window import Window
from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.uix.textinput import TextInput, TextInputCutCopyPaste
from kivy.uix.widget import Widget
from kivy.clock import Clock
class TextInputIMETest(unittest.TestCase):

    def test_ime(self):
        empty_ti = TextInput()
        empty_ti.focused = True
        ti = TextInput(text='abc')
        Window.dispatch('on_textedit', 'ㅎ')
        self.assertEqual(empty_ti.text, 'ㅎ')
        self.assertEqual(ti.text, 'abc')
        ti.focused = True
        Window.dispatch('on_textedit', 'ㅎ')
        self.assertEqual(ti.text, 'abcㅎ')
        Window.dispatch('on_textedit', '하')
        self.assertEqual(ti.text, 'abc하')
        Window.dispatch('on_textedit', '핫')
        Window.dispatch('on_textedit', '')
        Window.dispatch('on_textinput', '하')
        Window.dispatch('on_textedit', 'ㅅ')
        Window.dispatch('on_textedit', '세')
        self.assertEqual(ti.text, 'abc하세')