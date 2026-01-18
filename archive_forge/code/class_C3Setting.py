import unittest
class C3Setting:

    def __init__(self, setting, value):
        self._setting = setting
        self._value = value

    def __enter__(self):
        from zope.interface import ro
        setattr(ro.C3, self._setting.__name__, self._value)

    def __exit__(self, t, v, tb):
        from zope.interface import ro
        setattr(ro.C3, self._setting.__name__, self._setting)