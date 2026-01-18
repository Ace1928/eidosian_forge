import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
class __rl_safe_eval__:
    """creates one environment and re-uses it"""
    mode = 'eval'

    def __init__(self):
        self.env = None

    def __call__(self, expr, g=None, l=None, timeout=None, allowed_magic_methods=None):
        if not self.env:
            self.env = __RL_SAFE_ENV__(timeout=timeout, allowed_magic_methods=allowed_magic_methods)
        return self.env.__rl_safe_eval__(expr, g, l, self.mode, timeout=timeout, allowed_magic_methods=allowed_magic_methods, __frame_depth__=2)