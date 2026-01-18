import pytest
class TestRepetitiveSequence:

    def test_stop(self):
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100) + Animation(x=0)
        a.repeat = True
        w = Widget()
        a.start(w)
        a.stop(w)
        assert no_animations_being_played()

    def test_count_events(self, ec_cls):
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100, d=0.5) + Animation(x=0, d=0.5)
        a.repeat = True
        w = Widget()
        ec = ec_cls(a)
        ec1 = ec_cls(a.anim1)
        ec2 = ec_cls(a.anim2)
        a.start(w)
        ec.assert_(1, False, 0)
        ec1.assert_(1, False, 0)
        ec2.assert_(0, False, 0)
        sleep(0.2)
        ec.assert_(1, True, 0)
        ec1.assert_(1, True, 0)
        ec2.assert_(0, False, 0)
        sleep(0.5)
        ec.assert_(1, True, 0)
        ec1.assert_(1, True, 1)
        ec2.assert_(1, True, 0)
        sleep(0.5)
        ec.assert_(1, True, 0)
        ec1.assert_(2, True, 1)
        ec2.assert_(1, True, 1)
        sleep(0.5)
        ec.assert_(1, True, 0)
        ec1.assert_(2, True, 2)
        ec2.assert_(2, True, 1)
        a.stop(w)
        ec.assert_(1, True, 1)
        ec1.assert_(2, True, 2)
        ec2.assert_(2, True, 2)
        assert no_animations_being_played()