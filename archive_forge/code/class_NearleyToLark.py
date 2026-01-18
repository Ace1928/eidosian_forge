import os.path
import sys
import codecs
import argparse
from lark import Lark, Transformer, v_args
@v_args(inline=True)
class NearleyToLark(Transformer):

    def __init__(self):
        self._count = 0
        self.extra_rules = {}
        self.extra_rules_rev = {}
        self.alias_js_code = {}

    def _new_function(self, code):
        name = 'alias_%d' % self._count
        self._count += 1
        self.alias_js_code[name] = code
        return name

    def _extra_rule(self, rule):
        if rule in self.extra_rules_rev:
            return self.extra_rules_rev[rule]
        name = 'xrule_%d' % len(self.extra_rules)
        assert name not in self.extra_rules
        self.extra_rules[name] = rule
        self.extra_rules_rev[rule] = name
        return name

    def rule(self, name):
        return _get_rulename(name)

    def ruledef(self, name, exps):
        return '!%s: %s' % (_get_rulename(name), exps)

    def expr(self, item, op):
        rule = '(%s)%s' % (item, op)
        return self._extra_rule(rule)

    def regexp(self, r):
        return '/%s/' % r

    def null(self):
        return ''

    def string(self, s):
        return self._extra_rule(s)

    def expansion(self, *x):
        x, js = (x[:-1], x[-1])
        if js.children:
            js_code, = js.children
            js_code = js_code[2:-2]
            alias = '-> ' + self._new_function(js_code)
        else:
            alias = ''
        return ' '.join(x) + alias

    def expansions(self, *x):
        return '%s' % '\n    |'.join(x)

    def start(self, *rules):
        return '\n'.join(filter(None, rules))