from functools import partial
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.animation import Animation
from kivy.uix.stencilview import StencilView
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import BooleanProperty, OptionProperty, AliasProperty, \
class Carousel(StencilView):
    """Carousel class. See module documentation for more information.
    """
    slides = ListProperty([])
    'List of slides inside the Carousel. The slides are the\n    widgets added to the Carousel using the :attr:`add_widget` method.\n\n    :attr:`slides` is a :class:`~kivy.properties.ListProperty` and is\n    read-only.\n    '

    def _get_slides_container(self):
        return [x.parent for x in self.slides]
    slides_container = AliasProperty(_get_slides_container, bind=('slides',))
    direction = OptionProperty('right', options=('right', 'left', 'top', 'bottom'))
    "Specifies the direction in which the slides are ordered. This\n    corresponds to the direction from which the user swipes to go from one\n    slide to the next. It\n    can be `right`, `left`, `top`, or `bottom`. For example, with\n    the default value of `right`, the second slide is to the right\n    of the first and the user would swipe from the right towards the\n    left to get to the second slide.\n\n    :attr:`direction` is an :class:`~kivy.properties.OptionProperty` and\n    defaults to 'right'.\n    "
    min_move = NumericProperty(0.2)
    "Defines the minimum distance to be covered before the touch is\n    considered a swipe gesture and the Carousel content changed.\n    This is a expressed as a fraction of the Carousel's width.\n    If the movement doesn't reach this minimum value, the movement is\n    cancelled and the content is restored to its original position.\n\n    :attr:`min_move` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 0.2.\n    "
    anim_move_duration = NumericProperty(0.5)
    'Defines the duration of the Carousel animation between pages.\n\n    :attr:`anim_move_duration` is a :class:`~kivy.properties.NumericProperty`\n    and defaults to 0.5.\n    '
    anim_cancel_duration = NumericProperty(0.3)
    'Defines the duration of the animation when a swipe movement is not\n    accepted. This is generally when the user does not make a large enough\n    swipe. See :attr:`min_move`.\n\n    :attr:`anim_cancel_duration` is a :class:`~kivy.properties.NumericProperty`\n    and defaults to 0.3.\n    '
    loop = BooleanProperty(False)
    'Allow the Carousel to loop infinitely. If True, when the user tries to\n    swipe beyond last page, it will return to the first. If False, it will\n    remain on the last page.\n\n    :attr:`loop` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to False.\n    '

    def _get_index(self):
        if self.slides:
            return self._index % len(self.slides)
        return None

    def _set_index(self, value):
        if self.slides:
            self._index = value % len(self.slides)
        else:
            self._index = None
    index = AliasProperty(_get_index, _set_index, bind=('_index', 'slides'), cache=True)
    'Get/Set the current slide based on the index.\n\n    :attr:`index` is an :class:`~kivy.properties.AliasProperty` and defaults\n    to 0 (the first item).\n    '

    def _prev_slide(self):
        slides = self.slides
        len_slides = len(slides)
        index = self.index
        if len_slides < 2:
            return None
        if self.loop and index == 0:
            return slides[-1]
        if index > 0:
            return slides[index - 1]
    previous_slide = AliasProperty(_prev_slide, bind=('slides', 'index', 'loop'), cache=True)
    'The previous slide in the Carousel. It is None if the current slide is\n    the first slide in the Carousel. This ordering reflects the order in which\n    the slides are added: their presentation varies according to the\n    :attr:`direction` property.\n\n    :attr:`previous_slide` is an :class:`~kivy.properties.AliasProperty`.\n\n    .. versionchanged:: 1.5.0\n        This property no longer exposes the slides container. It returns\n        the widget you have added.\n    '

    def _curr_slide(self):
        if len(self.slides):
            return self.slides[self.index or 0]
    current_slide = AliasProperty(_curr_slide, bind=('slides', 'index'), cache=True)
    'The currently shown slide.\n\n    :attr:`current_slide` is an :class:`~kivy.properties.AliasProperty`.\n\n    .. versionchanged:: 1.5.0\n        The property no longer exposes the slides container. It returns\n        the widget you have added.\n    '

    def _next_slide(self):
        if len(self.slides) < 2:
            return None
        if self.loop and self.index == len(self.slides) - 1:
            return self.slides[0]
        if self.index < len(self.slides) - 1:
            return self.slides[self.index + 1]
    next_slide = AliasProperty(_next_slide, bind=('slides', 'index', 'loop'), cache=True)
    'The next slide in the Carousel. It is None if the current slide is\n    the last slide in the Carousel. This ordering reflects the order in which\n    the slides are added: their presentation varies according to the\n    :attr:`direction` property.\n\n    :attr:`next_slide` is an :class:`~kivy.properties.AliasProperty`.\n\n    .. versionchanged:: 1.5.0\n        The property no longer exposes the slides container.\n        It returns the widget you have added.\n    '
    scroll_timeout = NumericProperty(200)
    'Timeout allowed to trigger the :attr:`scroll_distance`, in milliseconds.\n    If the user has not moved :attr:`scroll_distance` within the timeout,\n    no scrolling will occur and the touch event will go to the children.\n\n    :attr:`scroll_timeout` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 200 (milliseconds)\n\n    .. versionadded:: 1.5.0\n    '
    scroll_distance = NumericProperty('20dp')
    "Distance to move before scrolling the :class:`Carousel` in pixels. As\n    soon as the distance has been traveled, the :class:`Carousel` will start\n    to scroll, and no touch event will go to children.\n    It is advisable that you base this value on the dpi of your target device's\n    screen.\n\n    :attr:`scroll_distance` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 20dp.\n\n    .. versionadded:: 1.5.0\n    "
    anim_type = StringProperty('out_quad')
    "Type of animation to use while animating to the next/previous slide.\n    This should be the name of an\n    :class:`~kivy.animation.AnimationTransition` function.\n\n    :attr:`anim_type` is a :class:`~kivy.properties.StringProperty` and\n    defaults to 'out_quad'.\n\n    .. versionadded:: 1.8.0\n    "
    ignore_perpendicular_swipes = BooleanProperty(False)
    'Ignore swipes on axis perpendicular to direction.\n\n    :attr:`ignore_perpendicular_swipes` is a\n    :class:`~kivy.properties.BooleanProperty` and defaults to False.\n\n    .. versionadded:: 1.10.0\n    '
    _index = NumericProperty(0, allownone=True)
    _prev = ObjectProperty(None, allownone=True)
    _current = ObjectProperty(None, allownone=True)
    _next = ObjectProperty(None, allownone=True)
    _offset = NumericProperty(0)
    _touch = ObjectProperty(None, allownone=True)
    _change_touch_mode_ev = None

    def __init__(self, **kwargs):
        self._trigger_position_visible_slides = Clock.create_trigger(self._position_visible_slides, -1)
        super(Carousel, self).__init__(**kwargs)
        self._skip_slide = None
        self.touch_mode_change = False
        self._prioritize_next = False
        self.fbind('loop', lambda *args: self._insert_visible_slides())

    def load_slide(self, slide):
        """Animate to the slide that is passed as the argument.

        .. versionchanged:: 1.8.0
        """
        slides = self.slides
        start, stop = (slides.index(self.current_slide), slides.index(slide))
        if start == stop:
            return
        self._skip_slide = stop
        if stop > start:
            self._prioritize_next = True
            self._insert_visible_slides(_next_slide=slide)
            self.load_next()
        else:
            self._prioritize_next = False
            self._insert_visible_slides(_prev_slide=slide)
            self.load_previous()

    def load_previous(self):
        """Animate to the previous slide.

        .. versionadded:: 1.7.0
        """
        self.load_next(mode='prev')

    def load_next(self, mode='next'):
        """Animate to the next slide.

        .. versionadded:: 1.7.0
        """
        if self.index is not None:
            w, h = self.size
            _direction = {'top': -h / 2, 'bottom': h / 2, 'left': w / 2, 'right': -w / 2}
            _offset = _direction[self.direction]
            if mode == 'prev':
                _offset = -_offset
            self._start_animation(min_move=0, offset=_offset)

    def get_slide_container(self, slide):
        return slide.parent

    @property
    def _prev_equals_next(self):
        return self.loop and len(self.slides) == 2

    def _insert_visible_slides(self, _next_slide=None, _prev_slide=None):
        get_slide_container = self.get_slide_container
        previous_slide = _prev_slide if _prev_slide else self.previous_slide
        if previous_slide:
            self._prev = get_slide_container(previous_slide)
        else:
            self._prev = None
        current_slide = self.current_slide
        if current_slide:
            self._current = get_slide_container(current_slide)
        else:
            self._current = None
        next_slide = _next_slide if _next_slide else self.next_slide
        if next_slide:
            self._next = get_slide_container(next_slide)
        else:
            self._next = None
        if self._prev_equals_next:
            setattr(self, '_prev' if self._prioritize_next else '_next', None)
        super_remove = super(Carousel, self).remove_widget
        for container in self.slides_container:
            super_remove(container)
        if self._prev and self._prev.parent is not self:
            super(Carousel, self).add_widget(self._prev)
        if self._next and self._next.parent is not self:
            super(Carousel, self).add_widget(self._next)
        if self._current:
            super(Carousel, self).add_widget(self._current)

    def _position_visible_slides(self, *args):
        slides, index = (self.slides, self.index)
        no_of_slides = len(slides) - 1
        if not slides:
            return
        x, y, width, height = (self.x, self.y, self.width, self.height)
        _offset, direction = (self._offset, self.direction[0])
        _prev, _next, _current = (self._prev, self._next, self._current)
        get_slide_container = self.get_slide_container
        last_slide = get_slide_container(slides[-1])
        first_slide = get_slide_container(slides[0])
        skip_next = False
        _loop = self.loop
        if direction in 'rl':
            xoff = x + _offset
            x_prev = {'l': xoff + width, 'r': xoff - width}
            x_next = {'l': xoff - width, 'r': xoff + width}
            if _prev:
                _prev.pos = (x_prev[direction], y)
            elif _loop and _next and (index == 0):
                if _offset > 0 and direction == 'r' or (_offset < 0 and direction == 'l'):
                    last_slide.pos = (x_prev[direction], y)
                    skip_next = True
            if _current:
                _current.pos = (xoff, y)
            if skip_next:
                return
            if _next:
                _next.pos = (x_next[direction], y)
            elif _loop and _prev and (index == no_of_slides):
                if _offset < 0 and direction == 'r' or (_offset > 0 and direction == 'l'):
                    first_slide.pos = (x_next[direction], y)
        if direction in 'tb':
            yoff = y + _offset
            y_prev = {'t': yoff - height, 'b': yoff + height}
            y_next = {'t': yoff + height, 'b': yoff - height}
            if _prev:
                _prev.pos = (x, y_prev[direction])
            elif _loop and _next and (index == 0):
                if _offset > 0 and direction == 't' or (_offset < 0 and direction == 'b'):
                    last_slide.pos = (x, y_prev[direction])
                    skip_next = True
            if _current:
                _current.pos = (x, yoff)
            if skip_next:
                return
            if _next:
                _next.pos = (x, y_next[direction])
            elif _loop and _prev and (index == no_of_slides):
                if _offset < 0 and direction == 't' or (_offset > 0 and direction == 'b'):
                    first_slide.pos = (x, y_next[direction])

    def on_size(self, *args):
        size = self.size
        for slide in self.slides_container:
            slide.size = size
        self._trigger_position_visible_slides()

    def on_pos(self, *args):
        self._trigger_position_visible_slides()

    def on_index(self, *args):
        self._insert_visible_slides()
        self._trigger_position_visible_slides()
        self._offset = 0

    def on_slides(self, *args):
        if self.slides:
            self.index = self.index % len(self.slides)
        self._insert_visible_slides()
        self._trigger_position_visible_slides()

    def on__offset(self, *args):
        self._trigger_position_visible_slides()
        direction = self.direction[0]
        _offset = self._offset
        width = self.width
        height = self.height
        index = self.index
        if self._skip_slide is not None or index is None:
            return
        if direction == 'r' and _offset <= -width or (direction == 'l' and _offset >= width) or (direction == 't' and _offset <= -height) or (direction == 'b' and _offset >= height):
            if self.next_slide:
                self.index += 1
        elif direction == 'r' and _offset >= width or (direction == 'l' and _offset <= -width) or (direction == 't' and _offset >= height) or (direction == 'b' and _offset <= -height):
            if self.previous_slide:
                self.index -= 1
        elif self._prev_equals_next:
            new_value = (_offset < 0) is (direction in 'rt')
            if self._prioritize_next is not new_value:
                self._prioritize_next = new_value
                if new_value is (self._next is None):
                    self._prev, self._next = (self._next, self._prev)

    def _start_animation(self, *args, **kwargs):
        new_offset = 0
        direction = kwargs.get('direction', self.direction)[0]
        is_horizontal = direction in 'rl'
        extent = self.width if is_horizontal else self.height
        min_move = kwargs.get('min_move', self.min_move)
        _offset = kwargs.get('offset', self._offset)
        if _offset < min_move * -extent:
            new_offset = -extent
        elif _offset > min_move * extent:
            new_offset = extent
        dur = self.anim_move_duration
        if new_offset == 0:
            dur = self.anim_cancel_duration
        len_slides = len(self.slides)
        index = self.index
        if not self.loop or len_slides == 1:
            is_first = index == 0
            is_last = index == len_slides - 1
            if direction in 'rt':
                towards_prev = new_offset > 0
                towards_next = new_offset < 0
            else:
                towards_prev = new_offset < 0
                towards_next = new_offset > 0
            if is_first and towards_prev or (is_last and towards_next):
                new_offset = 0
        anim = Animation(_offset=new_offset, d=dur, t=self.anim_type)
        anim.cancel_all(self)

        def _cmp(*l):
            if self._skip_slide is not None:
                self.index = self._skip_slide
                self._skip_slide = None
        anim.bind(on_complete=_cmp)
        anim.start(self)

    def _get_uid(self, prefix='sv'):
        return '{0}.{1}'.format(prefix, self.uid)

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            touch.ud[self._get_uid('cavoid')] = True
            return
        if self.disabled:
            return True
        if self._touch:
            return super(Carousel, self).on_touch_down(touch)
        Animation.cancel_all(self)
        self._touch = touch
        uid = self._get_uid()
        touch.grab(self)
        touch.ud[uid] = {'mode': 'unknown', 'time': touch.time_start}
        self._change_touch_mode_ev = Clock.schedule_once(self._change_touch_mode, self.scroll_timeout / 1000.0)
        self.touch_mode_change = False
        return True

    def on_touch_move(self, touch):
        if not self.touch_mode_change:
            if self.ignore_perpendicular_swipes and self.direction in ('top', 'bottom'):
                if abs(touch.oy - touch.y) < self.scroll_distance:
                    if abs(touch.ox - touch.x) > self.scroll_distance:
                        self._change_touch_mode()
                        self.touch_mode_change = True
            elif self.ignore_perpendicular_swipes and self.direction in ('right', 'left'):
                if abs(touch.ox - touch.x) < self.scroll_distance:
                    if abs(touch.oy - touch.y) > self.scroll_distance:
                        self._change_touch_mode()
                        self.touch_mode_change = True
        if self._get_uid('cavoid') in touch.ud:
            return
        if self._touch is not touch:
            super(Carousel, self).on_touch_move(touch)
            return self._get_uid() in touch.ud
        if touch.grab_current is not self:
            return True
        ud = touch.ud[self._get_uid()]
        direction = self.direction[0]
        if ud['mode'] == 'unknown':
            if direction in 'rl':
                distance = abs(touch.ox - touch.x)
            else:
                distance = abs(touch.oy - touch.y)
            if distance > self.scroll_distance:
                ev = self._change_touch_mode_ev
                if ev is not None:
                    ev.cancel()
                ud['mode'] = 'scroll'
        else:
            if direction in 'rl':
                self._offset += touch.dx
            if direction in 'tb':
                self._offset += touch.dy
        return True

    def on_touch_up(self, touch):
        if self._get_uid('cavoid') in touch.ud:
            return
        if self in [x() for x in touch.grab_list]:
            touch.ungrab(self)
            self._touch = None
            ud = touch.ud[self._get_uid()]
            if ud['mode'] == 'unknown':
                ev = self._change_touch_mode_ev
                if ev is not None:
                    ev.cancel()
                super(Carousel, self).on_touch_down(touch)
                Clock.schedule_once(partial(self._do_touch_up, touch), 0.1)
            else:
                self._start_animation()
        elif self._touch is not touch and self.uid not in touch.ud:
            super(Carousel, self).on_touch_up(touch)
        return self._get_uid() in touch.ud

    def _do_touch_up(self, touch, *largs):
        super(Carousel, self).on_touch_up(touch)
        for x in touch.grab_list[:]:
            touch.grab_list.remove(x)
            x = x()
            if not x:
                continue
            touch.grab_current = x
            super(Carousel, self).on_touch_up(touch)
        touch.grab_current = None

    def _change_touch_mode(self, *largs):
        if not self._touch:
            return
        self._start_animation()
        uid = self._get_uid()
        touch = self._touch
        ud = touch.ud[uid]
        if ud['mode'] == 'unknown':
            touch.ungrab(self)
            self._touch = None
            super(Carousel, self).on_touch_down(touch)
            return

    def add_widget(self, widget, index=0, *args, **kwargs):
        container = RelativeLayout(size=self.size, x=self.x - self.width, y=self.y)
        container.add_widget(widget)
        super(Carousel, self).add_widget(container, index, *args, **kwargs)
        if index != 0:
            self.slides.insert(index - len(self.slides), widget)
        else:
            self.slides.append(widget)

    def remove_widget(self, widget, *args, **kwargs):
        slides = self.slides
        if widget in slides:
            if self.index >= slides.index(widget):
                self.index = max(0, self.index - 1)
            container = widget.parent
            slides.remove(widget)
            super(Carousel, self).remove_widget(container, *args, **kwargs)
            container.remove_widget(widget)
            return
        super(Carousel, self).remove_widget(widget, *args, **kwargs)

    def clear_widgets(self, children=None, *args, **kwargs):
        if children is None:
            children = self.slides[:]
        remove_widget = self.remove_widget
        for widget in children:
            remove_widget(widget)
        super(Carousel, self).clear_widgets()