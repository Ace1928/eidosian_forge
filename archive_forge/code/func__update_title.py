from kivy.animation import Animation
from kivy.uix.floatlayout import FloatLayout
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import (ObjectProperty, StringProperty,
from kivy.uix.widget import Widget
from kivy.logger import Logger
def _update_title(self, dt):
    if not self.container_title:
        self._trigger_title()
        return
    c = self.container_title
    c.clear_widgets()
    instance = Builder.template(self.title_template, title=self.title, item=self, **self.title_args)
    c.add_widget(instance)