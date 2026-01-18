import sys
from os import environ
from os.path import join
from copy import copy
from types import CodeType
from functools import partial
from kivy.factory import Factory
from kivy.lang.parser import (
from kivy.logger import Logger
from kivy.utils import QueryDict
from kivy.cache import Cache
from kivy import kivy_data_dir
from kivy.context import register_context
from kivy.resources import resource_find
from kivy._event import Observable, EventDispatcher
def _build_canvas(self, canvas, widget, rule, rootrule):
    global Instruction
    if Instruction is None:
        Instruction = Factory.get('Instruction')
    idmap = copy(self.rulectx[rootrule]['ids'])
    for crule in rule.children:
        name = crule.name
        if name == 'Clear':
            canvas.clear()
            continue
        instr = Factory.get(name)()
        if not isinstance(instr, Instruction):
            raise BuilderException(crule.ctx, crule.line, 'You can add only graphics Instruction in canvas.')
        try:
            for prule in crule.properties.values():
                key = prule.name
                value = prule.co_value
                if type(value) is CodeType:
                    value, _ = create_handler(widget, instr.proxy_ref, key, value, prule, idmap, True)
                setattr(instr, key, value)
        except Exception as e:
            tb = sys.exc_info()[2]
            raise BuilderException(prule.ctx, prule.line, '{}: {}'.format(e.__class__.__name__, e), cause=tb)