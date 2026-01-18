import re
from functools import partial
from tornado import httputil
from tornado.httpserver import _CallableAdapter
from tornado.escape import url_escape, url_unescape, utf8
from tornado.log import app_log
from tornado.util import basestring_type, import_object, re_unescape, unicode_type
from typing import Any, Union, Optional, Awaitable, List, Dict, Pattern, Tuple, overload
class ReversibleRuleRouter(ReversibleRouter, RuleRouter):
    """A rule-based router that implements ``reverse_url`` method.

    Each rule added to this router may have a ``name`` attribute that can be
    used to reconstruct an original uri. The actual reconstruction takes place
    in a rule's matcher (see `Matcher.reverse`).
    """

    def __init__(self, rules: Optional[_RuleList]=None) -> None:
        self.named_rules = {}
        super().__init__(rules)

    def process_rule(self, rule: 'Rule') -> 'Rule':
        rule = super().process_rule(rule)
        if rule.name:
            if rule.name in self.named_rules:
                app_log.warning('Multiple handlers named %s; replacing previous value', rule.name)
            self.named_rules[rule.name] = rule
        return rule

    def reverse_url(self, name: str, *args: Any) -> Optional[str]:
        if name in self.named_rules:
            return self.named_rules[name].matcher.reverse(*args)
        for rule in self.rules:
            if isinstance(rule.target, ReversibleRouter):
                reversed_url = rule.target.reverse_url(name, *args)
                if reversed_url is not None:
                    return reversed_url
        return None