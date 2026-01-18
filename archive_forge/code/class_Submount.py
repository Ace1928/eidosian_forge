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
class Submount(RuleFactory):
    """Like `Subdomain` but prefixes the URL rule with a given string::

        url_map = Map([
            Rule('/', endpoint='index'),
            Submount('/blog', [
                Rule('/', endpoint='blog/index'),
                Rule('/entry/<entry_slug>', endpoint='blog/show')
            ])
        ])

    Now the rule ``'blog/show'`` matches ``/blog/entry/<entry_slug>``.
    """

    def __init__(self, path: str, rules: t.Iterable[RuleFactory]) -> None:
        self.path = path.rstrip('/')
        self.rules = rules

    def get_rules(self, map: Map) -> t.Iterator[Rule]:
        for rulefactory in self.rules:
            for rule in rulefactory.get_rules(map):
                rule = rule.empty()
                rule.rule = self.path + rule.rule
                yield rule