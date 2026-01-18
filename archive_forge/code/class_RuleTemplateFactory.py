from __future__ import annotations
import ast
import re
import typing as t
from dataclasses import dataclass
from string import Template
from types import CodeType
from urllib.parse import quote
from ..datastructures import iter_multi_items
from ..urls import _urlencode
from .converters import ValidationError
class RuleTemplateFactory(RuleFactory):
    """A factory that fills in template variables into rules.  Used by
    `RuleTemplate` internally.

    :internal:
    """

    def __init__(self, rules: t.Iterable[RuleFactory], context: dict[str, t.Any]) -> None:
        self.rules = rules
        self.context = context

    def get_rules(self, map: Map) -> t.Iterator[Rule]:
        for rulefactory in self.rules:
            for rule in rulefactory.get_rules(map):
                new_defaults = subdomain = None
                if rule.defaults:
                    new_defaults = {}
                    for key, value in rule.defaults.items():
                        if isinstance(value, str):
                            value = Template(value).substitute(self.context)
                        new_defaults[key] = value
                if rule.subdomain is not None:
                    subdomain = Template(rule.subdomain).substitute(self.context)
                new_endpoint = rule.endpoint
                if isinstance(new_endpoint, str):
                    new_endpoint = Template(new_endpoint).substitute(self.context)
                yield Rule(Template(rule.rule).substitute(self.context), new_defaults, subdomain, rule.methods, rule.build_only, new_endpoint, rule.strict_slashes)