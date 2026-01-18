from kivy.animation import Animation
from kivy.uix.floatlayout import FloatLayout
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import (ObjectProperty, StringProperty,
from kivy.uix.widget import Widget
from kivy.logger import Logger
def _do_layout(self, dt):
    children = self.children
    if children:
        all_collapsed = all((x.collapse for x in children))
    else:
        all_collapsed = False
    if all_collapsed:
        children[0].collapse = False
    orientation = self.orientation
    min_space = self.min_space
    min_space_total = len(children) * self.min_space
    w, h = self.size
    x, y = self.pos
    if orientation == 'horizontal':
        display_space = self.width - min_space_total
    else:
        display_space = self.height - min_space_total
    if display_space <= 0:
        Logger.warning('Accordion: not enough space for displaying all children')
        Logger.warning('Accordion: need %dpx, got %dpx' % (min_space_total, min_space_total + display_space))
        Logger.warning('Accordion: layout aborted.')
        return
    if orientation == 'horizontal':
        children = reversed(children)
    for child in children:
        child_space = min_space
        child_space += display_space * (1 - child.collapse_alpha)
        child._min_space = min_space
        child.x = x
        child.y = y
        child.orientation = self.orientation
        if orientation == 'horizontal':
            child.content_size = (display_space, h)
            child.width = child_space
            child.height = h
            x += child_space
        else:
            child.content_size = (w, display_space)
            child.width = w
            child.height = child_space
            y += child_space