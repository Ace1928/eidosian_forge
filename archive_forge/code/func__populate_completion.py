import sys
import os
import time
import locale
import signal
import urwid
from typing import Optional
from . import args as bpargs, repl, translations
from .formatter import theme_map
from .translations import _
from .keys import urwid_key_dispatch as key_dispatch
def _populate_completion(self):
    widget_list = self.tooltip.body
    while widget_list:
        widget_list.pop()
    if self.complete():
        if self.funcprops:
            func_name = self.funcprops.func
            args = self.funcprops.argspec.args
            is_bound = self.funcprops.is_bound_method
            in_arg = self.arg_pos
            varargs = self.funcprops.argspec.varargs
            varkw = self.funcprops.argspec.varkwargs
            defaults = self.funcprops.argspec.defaults
            kwonly = self.funcprops.argspec.kwonly
            kwonly_defaults = self.funcprops.argspec.kwonly_defaults or {}
            markup = [('bold name', func_name), ('name', ': (')]
            if is_bound and isinstance(in_arg, int):
                in_arg += 1
            for k, i in enumerate(args):
                if defaults and k + 1 > len(args) - len(defaults):
                    kw = repr(defaults[k - (len(args) - len(defaults))])
                else:
                    kw = None
                if not k and str(i) == 'self':
                    color = 'name'
                else:
                    color = 'token'
                if k == in_arg or i == in_arg:
                    color = 'bold ' + color
                markup.append((color, str(i)))
                if kw is not None:
                    markup.extend([('punctuation', '='), ('token', kw)])
                if k != len(args) - 1:
                    markup.append(('punctuation', ', '))
            if varargs:
                if args:
                    markup.append(('punctuation', ', '))
                markup.append(('token', '*' + varargs))
            if kwonly:
                if not varargs:
                    if args:
                        markup.append(('punctuation', ', '))
                    markup.append(('punctuation', '*'))
                for arg in kwonly:
                    if arg == in_arg:
                        color = 'bold token'
                    else:
                        color = 'token'
                    markup.extend([('punctuation', ', '), (color, arg)])
                    if arg in kwonly_defaults:
                        markup.extend([('punctuation', '='), ('token', repr(kwonly_defaults[arg]))])
            if varkw:
                if args or varargs or kwonly:
                    markup.append(('punctuation', ', '))
                markup.append(('token', '**' + varkw))
            markup.append(('punctuation', ')'))
            widget_list.append(urwid.Text(markup))
        if self.matches_iter.matches:
            attr_map = {}
            focus_map = {'main': 'operator'}
            texts = [urwid.AttrMap(urwid.Text(('main', match)), attr_map, focus_map) for match in self.matches_iter.matches]
            width = max((text.original_widget.pack()[0] for text in texts))
            gridflow = urwid.GridFlow(texts, width, 1, 0, 'left')
            widget_list.append(gridflow)
            self.tooltip.grid = gridflow
            self.overlay.tooltip_focus = False
        else:
            self.tooltip.grid = None
        self.frame.body = self.overlay
    else:
        self.frame.body = self.listbox
        self.tooltip.grid = None
    if self.docstring:
        docstring = self.docstring
        widget_list.append(urwid.Text(('comment', docstring)))