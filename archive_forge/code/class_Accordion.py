from kivy.animation import Animation
from kivy.uix.floatlayout import FloatLayout
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import (ObjectProperty, StringProperty,
from kivy.uix.widget import Widget
from kivy.logger import Logger
class Accordion(Widget):
    """Accordion class. See module documentation for more information.
    """
    orientation = OptionProperty('horizontal', options=('horizontal', 'vertical'))
    "Orientation of the layout.\n\n    :attr:`orientation` is an :class:`~kivy.properties.OptionProperty`\n    and defaults to 'horizontal'. Can take a value of 'vertical' or\n    'horizontal'.\n\n    "
    anim_duration = NumericProperty(0.25)
    'Duration of the animation in seconds when a new accordion item is\n    selected.\n\n    :attr:`anim_duration` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to .25 (250ms).\n    '
    anim_func = ObjectProperty('out_expo')
    "Easing function to use for the animation. Check\n    :class:`kivy.animation.AnimationTransition` for more information about\n    available animation functions.\n\n    :attr:`anim_func` is an :class:`~kivy.properties.ObjectProperty` and\n    defaults to 'out_expo'. You can set a string or a function to use as an\n    easing function.\n    "
    min_space = NumericProperty('44dp')
    'Minimum space to use for the title of each item. This value is\n    automatically set for each child every time the layout event occurs.\n\n    :attr:`min_space` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 44 (px).\n    '

    def __init__(self, **kwargs):
        super(Accordion, self).__init__(**kwargs)
        update = self._trigger_layout = Clock.create_trigger(self._do_layout, -1)
        fbind = self.fbind
        fbind('orientation', update)
        fbind('children', update)
        fbind('size', update)
        fbind('pos', update)
        fbind('min_space', update)

    def add_widget(self, widget, *args, **kwargs):
        if not isinstance(widget, AccordionItem):
            raise AccordionException('Accordion accept only AccordionItem')
        widget.accordion = self
        super(Accordion, self).add_widget(widget, *args, **kwargs)

    def select(self, instance):
        if instance not in self.children:
            raise AccordionException('Accordion: instance not found in children')
        for widget in self.children:
            widget.collapse = widget is not instance
        self._trigger_layout()

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