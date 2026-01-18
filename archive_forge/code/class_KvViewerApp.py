from sys import argv
from kivy.lang import Builder
from kivy.app import App
from kivy.core.window import Window
from kivy.clock import Clock, mainthread
from kivy.uix.label import Label
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from os.path import dirname, basename, join
class KvViewerApp(App):

    def build(self):
        o = Observer()
        o.schedule(KvHandler(self.update, TARGET), PATH)
        o.start()
        Clock.schedule_once(self.update, 1)
        return super(KvViewerApp, self).build()

    @mainthread
    def update(self, *args):
        Builder.unload_file(join(PATH, TARGET))
        for w in Window.children[:]:
            Window.remove_widget(w)
        try:
            Window.add_widget(Builder.load_file(join(PATH, TARGET)))
        except Exception as e:
            Window.add_widget(Label(text=e.message if getattr(e, 'message', None) else str(e)))