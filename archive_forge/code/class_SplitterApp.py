from kivy.factory import Factory
from kivy.uix.button import Button
from kivy.properties import (OptionProperty, NumericProperty, ObjectProperty,
from kivy.uix.boxlayout import BoxLayout
class SplitterApp(App):

    def build(self):
        root = FloatLayout()
        bx = BoxLayout()
        bx.add_widget(Button())
        bx.add_widget(Button())
        bx2 = BoxLayout()
        bx2.add_widget(Button())
        bx2.add_widget(Button())
        bx2.add_widget(Button())
        spl = Splitter(size_hint=(1, 0.25), pos_hint={'top': 1}, sizable_from='bottom')
        spl1 = Splitter(sizable_from='left', size_hint=(None, 1), width=90)
        spl1.add_widget(Button())
        bx.add_widget(spl1)
        spl.add_widget(bx)
        spl2 = Splitter(size_hint=(0.25, 1))
        spl2.add_widget(bx2)
        spl2.sizable_from = 'right'
        root.add_widget(spl)
        root.add_widget(spl2)
        return root