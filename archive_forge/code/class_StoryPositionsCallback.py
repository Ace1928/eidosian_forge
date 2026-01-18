from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
class StoryPositionsCallback(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self):
        if self.__class__ == StoryPositionsCallback:
            _self = None
        else:
            _self = self
        _mupdf.StoryPositionsCallback_swiginit(self, _mupdf.new_StoryPositionsCallback(_self))

    def call(self, position):
        return _mupdf.StoryPositionsCallback_call(self, position)

    @staticmethod
    def s_call(ctx, self0, position):
        return _mupdf.StoryPositionsCallback_s_call(ctx, self0, position)
    __swig_destroy__ = _mupdf.delete_StoryPositionsCallback

    def __disown__(self):
        self.this.disown()
        _mupdf.disown_StoryPositionsCallback(self)
        return weakref.proxy(self)