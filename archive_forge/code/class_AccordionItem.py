from kivy.animation import Animation
from kivy.uix.floatlayout import FloatLayout
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import (ObjectProperty, StringProperty,
from kivy.uix.widget import Widget
from kivy.logger import Logger
class AccordionItem(FloatLayout):
    """AccordionItem class that must be used in conjunction with the
    :class:`Accordion` class. See the module documentation for more
    information.
    """
    title = StringProperty('')
    "Title string of the item. The title might be used in conjunction with the\n    `AccordionItemTitle` template. If you are using a custom template, you can\n    use that property as a text entry, or not. By default, it's used for the\n    title text. See title_template and the example below.\n\n    :attr:`title` is a :class:`~kivy.properties.StringProperty` and defaults\n    to ''.\n    "
    title_template = StringProperty('AccordionItemTitle')
    "Template to use for creating the title part of the accordion item. The\n    default template is a simple Label, not customizable (except the text) that\n    supports vertical and horizontal orientation and different backgrounds for\n    collapse and selected mode.\n\n    It's better to create and use your own template if the default template\n    does not suffice.\n\n    :attr:`title` is a :class:`~kivy.properties.StringProperty` and defaults to\n    'AccordionItemTitle'. The current default template lives in the\n    `kivy/data/style.kv` file.\n\n    Here is the code if you want to build your own template::\n\n        [AccordionItemTitle@Label]:\n            text: ctx.title\n            canvas.before:\n                Color:\n                    rgb: 1, 1, 1\n                BorderImage:\n                    source:\n                        ctx.item.background_normal                         if ctx.item.collapse                         else ctx.item.background_selected\n                    pos: self.pos\n                    size: self.size\n                PushMatrix\n                Translate:\n                    xy: self.center_x, self.center_y\n                Rotate:\n                    angle: 90 if ctx.item.orientation == 'horizontal' else 0\n                    axis: 0, 0, 1\n                Translate:\n                    xy: -self.center_x, -self.center_y\n            canvas.after:\n                PopMatrix\n\n\n    "
    title_args = DictProperty({})
    'Default arguments that will be passed to the\n    :meth:`kivy.lang.Builder.template` method.\n\n    :attr:`title_args` is a :class:`~kivy.properties.DictProperty` and defaults\n    to {}.\n    '
    collapse = BooleanProperty(True)
    'Boolean to indicate if the current item is collapsed or not.\n\n    :attr:`collapse` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to True.\n    '
    collapse_alpha = NumericProperty(1.0)
    "Value between 0 and 1 to indicate how much the item is collapsed (1) or\n    whether it is selected (0). It's mostly used for animation.\n\n    :attr:`collapse_alpha` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 1.\n    "
    accordion = ObjectProperty(None)
    'Instance of the :class:`Accordion` that the item belongs to.\n\n    :attr:`accordion` is an :class:`~kivy.properties.ObjectProperty` and\n    defaults to None.\n    '
    background_normal = StringProperty('atlas://data/images/defaulttheme/button')
    "Background image of the accordion item used for the default graphical\n    representation when the item is collapsed.\n\n    :attr:`background_normal` is a :class:`~kivy.properties.StringProperty` and\n    defaults to 'atlas://data/images/defaulttheme/button'.\n    "
    background_disabled_normal = StringProperty('atlas://data/images/defaulttheme/button_disabled')
    "Background image of the accordion item used for the default graphical\n    representation when the item is collapsed and disabled.\n\n    .. versionadded:: 1.8.0\n\n    :attr:`background__disabled_normal` is a\n    :class:`~kivy.properties.StringProperty` and defaults to\n    'atlas://data/images/defaulttheme/button_disabled'.\n    "
    background_selected = StringProperty('atlas://data/images/defaulttheme/button_pressed')
    "Background image of the accordion item used for the default graphical\n    representation when the item is selected (not collapsed).\n\n    :attr:`background_normal` is a :class:`~kivy.properties.StringProperty` and\n    defaults to 'atlas://data/images/defaulttheme/button_pressed'.\n    "
    background_disabled_selected = StringProperty('atlas://data/images/defaulttheme/button_disabled_pressed')
    "Background image of the accordion item used for the default graphical\n    representation when the item is selected (not collapsed) and disabled.\n\n    .. versionadded:: 1.8.0\n\n    :attr:`background_disabled_selected` is a\n    :class:`~kivy.properties.StringProperty` and defaults to\n    'atlas://data/images/defaulttheme/button_disabled_pressed'.\n    "
    orientation = OptionProperty('vertical', options=('horizontal', 'vertical'))
    'Link to the :attr:`Accordion.orientation` property.\n    '
    min_space = NumericProperty('44dp')
    'Link to the :attr:`Accordion.min_space` property.\n    '
    content_size = ListProperty([100, 100])
    '(internal) Set by the :class:`Accordion` to the size allocated for the\n    content.\n    '
    container = ObjectProperty(None)
    '(internal) Property that will be set to the container of children inside\n    the AccordionItem representation.\n    '
    container_title = ObjectProperty(None)
    '(internal) Property that will be set to the container of title inside\n    the AccordionItem representation.\n    '

    def __init__(self, **kwargs):
        self._trigger_title = Clock.create_trigger(self._update_title, -1)
        self._anim_collapse = None
        super(AccordionItem, self).__init__(**kwargs)
        trigger_title = self._trigger_title
        fbind = self.fbind
        fbind('title', trigger_title)
        fbind('title_template', trigger_title)
        fbind('title_args', trigger_title)
        trigger_title()

    def add_widget(self, *args, **kwargs):
        if self.container is None:
            super(AccordionItem, self).add_widget(*args, **kwargs)
            return
        self.container.add_widget(*args, **kwargs)

    def remove_widget(self, *args, **kwargs):
        if self.container:
            self.container.remove_widget(*args, **kwargs)
            return
        super(AccordionItem, self).remove_widget(*args, **kwargs)

    def on_collapse(self, instance, value):
        accordion = self.accordion
        if accordion is None:
            return
        if not value:
            self.accordion.select(self)
        collapse_alpha = float(value)
        if self._anim_collapse:
            self._anim_collapse.stop(self)
            self._anim_collapse = None
        if self.collapse_alpha != collapse_alpha:
            self._anim_collapse = Animation(collapse_alpha=collapse_alpha, t=accordion.anim_func, d=accordion.anim_duration)
            self._anim_collapse.start(self)

    def on_collapse_alpha(self, instance, value):
        self.accordion._trigger_layout()

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return
        if self.disabled:
            return True
        if self.collapse:
            self.collapse = False
            return True
        else:
            return super(AccordionItem, self).on_touch_down(touch)

    def _update_title(self, dt):
        if not self.container_title:
            self._trigger_title()
            return
        c = self.container_title
        c.clear_widgets()
        instance = Builder.template(self.title_template, title=self.title, item=self, **self.title_args)
        c.add_widget(instance)