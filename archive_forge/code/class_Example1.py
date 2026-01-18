from functools import partial
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.animation import Animation
from kivy.uix.stencilview import StencilView
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import BooleanProperty, OptionProperty, AliasProperty, \
class Example1(App):

    def build(self):
        carousel = Carousel(direction='left', loop=True)
        for i in range(4):
            src = 'http://placehold.it/480x270.png&text=slide-%d&.png' % i
            image = Factory.AsyncImage(source=src, fit_mode='contain')
            carousel.add_widget(image)
        return carousel