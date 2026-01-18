import pytest
class TestAnimation:

    def test_start_animation(self):
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100, d=1)
        w = Widget()
        a.start(w)
        sleep(1.5)
        assert w.x == pytest.approx(100)
        assert no_animations_being_played()

    def test_animation_duration_0(self):
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100, d=0)
        w = Widget()
        a.start(w)
        sleep(0.5)
        assert no_animations_being_played()

    def test_cancel_all(self):
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a1 = Animation(x=100)
        a2 = Animation(y=100)
        w1 = Widget()
        w2 = Widget()
        a1.start(w1)
        a1.start(w2)
        a2.start(w1)
        a2.start(w2)
        assert not no_animations_being_played()
        Animation.cancel_all(None)
        assert no_animations_being_played()

    def test_cancel_all_2(self):
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a1 = Animation(x=100)
        a2 = Animation(y=100)
        w1 = Widget()
        w2 = Widget()
        a1.start(w1)
        a1.start(w2)
        a2.start(w1)
        a2.start(w2)
        assert not no_animations_being_played()
        Animation.cancel_all(None, 'x', 'z')
        assert not no_animations_being_played()
        Animation.cancel_all(None, 'y')
        assert no_animations_being_played()

    def test_stop_animation(self):
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100, d=1)
        w = Widget()
        a.start(w)
        sleep(0.5)
        a.stop(w)
        assert w.x != pytest.approx(100)
        assert w.x != pytest.approx(0)
        assert no_animations_being_played()

    def test_stop_all(self):
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100, d=1)
        w = Widget()
        a.start(w)
        sleep(0.5)
        Animation.stop_all(w)
        assert no_animations_being_played()

    def test_stop_all_2(self):
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100, d=1)
        w = Widget()
        a.start(w)
        sleep(0.5)
        Animation.stop_all(w, 'x')
        assert no_animations_being_played()

    def test_duration(self):
        from kivy.animation import Animation
        a = Animation(x=100, d=1)
        assert a.duration == 1

    def test_transition(self):
        from kivy.animation import Animation, AnimationTransition
        a = Animation(x=100, t='out_bounce')
        assert a.transition is AnimationTransition.out_bounce

    def test_animated_properties(self):
        from kivy.animation import Animation
        a = Animation(x=100)
        assert a.animated_properties == {'x': 100}

    def test_animated_instruction(self):
        from kivy.graphics import Scale
        from kivy.animation import Animation
        a = Animation(x=100, d=1)
        instruction = Scale(3, 3, 3)
        a.start(instruction)
        assert a.animated_properties == {'x': 100}
        assert instruction.x == pytest.approx(3)
        sleep(1.5)
        assert instruction.x == pytest.approx(100)
        assert no_animations_being_played()

    def test_weakref(self):
        import gc
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        w = Widget()
        a = Animation(x=100)
        a.start(w.proxy_ref)
        del w
        gc.collect()
        try:
            sleep(1.0)
        except ReferenceError:
            pass
        assert no_animations_being_played()