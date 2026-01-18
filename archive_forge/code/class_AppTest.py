from os import name
import os.path
from math import isclose
from textwrap import dedent
from kivy.app import App
from kivy.clock import Clock
from kivy import lang
from kivy.tests import GraphicUnitTest, async_run, UnitKivyApp
class AppTest(GraphicUnitTest):

    def test_start_raw_app(self):
        lang._delayed_start = None
        a = App()
        Clock.schedule_once(a.stop, 0.1)
        a.run()

    def test_start_app_with_kv(self):

        class TestKvApp(App):
            pass
        lang._delayed_start = None
        a = TestKvApp()
        Clock.schedule_once(a.stop, 0.1)
        a.run()

    def test_user_data_dir(self):
        a = App()
        data_dir = a.user_data_dir
        assert os.path.exists(data_dir)

    def test_directory(self):
        a = App()
        assert os.path.exists(a.directory)

    def test_name(self):

        class NameTest(App):
            pass
        a = NameTest()
        assert a.name == 'nametest'