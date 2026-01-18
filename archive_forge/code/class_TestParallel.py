import pytest
class TestParallel:

    def test_have_properties_to_animate(self):
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100) & Animation(y=100)
        w = Widget()
        assert not a.have_properties_to_animate(w)
        a.start(w)
        assert a.have_properties_to_animate(w)
        a.stop(w)
        assert not a.have_properties_to_animate(w)
        assert no_animations_being_played()

    def test_cancel_property(self):
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100) & Animation(y=100)
        w = Widget()
        a.start(w)
        a.cancel_property(w, 'x')
        assert not no_animations_being_played()
        a.stop(w)
        assert no_animations_being_played()

    def test_animated_properties(self):
        from kivy.animation import Animation
        a = Animation(x=100) & Animation(y=100)
        assert a.animated_properties == {'x': 100, 'y': 100}

    def test_transition(self):
        from kivy.animation import Animation
        a = Animation(x=100) & Animation(y=100)
        with pytest.raises(AttributeError):
            a.transition

    def test_count_events(self, ec_cls):
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100) & Animation(y=100, d=0.5)
        w = Widget()
        ec = ec_cls(a)
        ec1 = ec_cls(a.anim1)
        ec2 = ec_cls(a.anim2)
        a.start(w)
        ec.assert_(1, False, 0)
        ec1.assert_(1, False, 0)
        ec2.assert_(1, False, 0)
        sleep(0.2)
        ec.assert_(1, False, 0)
        ec1.assert_(1, True, 0)
        ec2.assert_(1, True, 0)
        sleep(0.5)
        ec.assert_(1, False, 0)
        ec1.assert_(1, True, 0)
        ec2.assert_(1, True, 1)
        sleep(0.5)
        ec.assert_(1, False, 1)
        ec1.assert_(1, True, 1)
        ec2.assert_(1, True, 1)
        assert no_animations_being_played()