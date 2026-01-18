from kivy.compat import iteritems
from kivy.logger import Logger
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import (StringProperty, ObjectProperty, AliasProperty,
from kivy.animation import Animation, AnimationTransition
from kivy.uix.relativelayout import RelativeLayout
from kivy.lang import Builder
from kivy.graphics import (RenderContext, Rectangle, Fbo,
class TransitionBase(EventDispatcher):
    """TransitionBase is used to animate 2 screens within the
    :class:`ScreenManager`. This class acts as a base for other
    implementations like the :class:`SlideTransition` and
    :class:`SwapTransition`.

    :Events:
        `on_progress`: Transition object, progression float
            Fired during the animation of the transition.
        `on_complete`: Transition object
            Fired when the transition is finished.
    """
    screen_out = ObjectProperty()
    'Property that contains the screen to hide.\n    Automatically set by the :class:`ScreenManager`.\n\n    :class:`screen_out` is an :class:`~kivy.properties.ObjectProperty` and\n    defaults to None.\n    '
    screen_in = ObjectProperty()
    'Property that contains the screen to show.\n    Automatically set by the :class:`ScreenManager`.\n\n    :class:`screen_in` is an :class:`~kivy.properties.ObjectProperty` and\n    defaults to None.\n    '
    duration = NumericProperty(0.4)
    'Duration in seconds of the transition.\n\n    :class:`duration` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to .4 (= 400ms).\n\n    .. versionchanged:: 1.8.0\n\n        Default duration has been changed from 700ms to 400ms.\n    '
    manager = ObjectProperty()
    ':class:`ScreenManager` object, set when the screen is added to a\n    manager.\n\n    :attr:`manager` is an :class:`~kivy.properties.ObjectProperty` and\n    defaults to None, read-only.\n\n    '
    is_active = BooleanProperty(False)
    'Indicate whether the transition is currently active or not.\n\n    :attr:`is_active` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to False, read-only.\n    '
    _anim = ObjectProperty(allownone=True)
    __events__ = ('on_progress', 'on_complete')

    def start(self, manager):
        """(internal) Starts the transition. This is automatically
        called by the :class:`ScreenManager`.
        """
        if self.is_active:
            raise ScreenManagerException('start() is called twice!')
        self.manager = manager
        self._anim = Animation(d=self.duration, s=0)
        self._anim.bind(on_progress=self._on_progress, on_complete=self._on_complete)
        self.add_screen(self.screen_in)
        self.screen_in.transition_progress = 0.0
        self.screen_in.transition_state = 'in'
        self.screen_out.transition_progress = 0.0
        self.screen_out.transition_state = 'out'
        self.screen_in.dispatch('on_pre_enter')
        self.screen_out.dispatch('on_pre_leave')
        self.is_active = True
        self._anim.start(self)
        self.dispatch('on_progress', 0)

    def stop(self):
        """(internal) Stops the transition. This is automatically called by the
        :class:`ScreenManager`.
        """
        if self._anim:
            self._anim.cancel(self)
            self.dispatch('on_complete')
            self._anim = None
        self.is_active = False

    def add_screen(self, screen):
        """(internal) Used to add a screen to the :class:`ScreenManager`.
        """
        self.manager.real_add_widget(screen)

    def remove_screen(self, screen):
        """(internal) Used to remove a screen from the :class:`ScreenManager`.
        """
        self.manager.real_remove_widget(screen)

    def on_complete(self):
        self.remove_screen(self.screen_out)

    def on_progress(self, progression):
        pass

    def _on_progress(self, *l):
        progress = l[-1]
        self.screen_in.transition_progress = progress
        self.screen_out.transition_progress = 1.0 - progress
        self.dispatch('on_progress', progress)

    def _on_complete(self, *l):
        self.is_active = False
        self.dispatch('on_complete')
        self.screen_in.dispatch('on_enter')
        self.screen_out.dispatch('on_leave')
        self._anim = None