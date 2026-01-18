import unittest
import os
from weakref import proxy
from functools import partial
from textwrap import dedent
class BaseClass(object):
    uid = 0

    def __init__(self, **kwargs):
        super(BaseClass, self).__init__()
        self.proxy_ref = proxy(self)
        self.children = []
        self.parent = None
        self.binded_func = {}
        self.id = None
        self.ids = {}
        self.cls = []
        self.ids = {}
        self.uid = BaseClass.uid
        BaseClass.uid += 1

    def add_widget(self, widget):
        self.children.append(widget)
        widget.parent = self

    def dispatch(self, event_type, *largs, **kwargs):
        pass

    def create_property(self, name, value=None, default_value=True):
        pass

    def is_event_type(self, key):
        return key.startswith('on_')

    def fbind(self, name, func, *largs):
        self.binded_func[name] = partial(func, *largs)
        return True

    def apply_class_lang_rules(self, root=None, ignored_consts=set(), rule_children=None):
        pass