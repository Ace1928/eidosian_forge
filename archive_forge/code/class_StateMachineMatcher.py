from __future__ import annotations
import re
import typing as t
from dataclasses import dataclass
from dataclasses import field
from .converters import ValidationError
from .exceptions import NoMatch
from .exceptions import RequestAliasRedirect
from .exceptions import RequestPath
from .rules import Rule
from .rules import RulePart
class StateMachineMatcher:

    def __init__(self, merge_slashes: bool) -> None:
        self._root = State()
        self.merge_slashes = merge_slashes

    def add(self, rule: Rule) -> None:
        state = self._root
        for part in rule._parts:
            if part.static:
                state.static.setdefault(part.content, State())
                state = state.static[part.content]
            else:
                for test_part, new_state in state.dynamic:
                    if test_part == part:
                        state = new_state
                        break
                else:
                    new_state = State()
                    state.dynamic.append((part, new_state))
                    state = new_state
        state.rules.append(rule)

    def update(self) -> None:
        state = self._root

        def _update_state(state: State) -> None:
            state.dynamic.sort(key=lambda entry: entry[0].weight)
            for new_state in state.static.values():
                _update_state(new_state)
            for _, new_state in state.dynamic:
                _update_state(new_state)
        _update_state(state)

    def match(self, domain: str, path: str, method: str, websocket: bool) -> tuple[Rule, t.MutableMapping[str, t.Any]]:
        have_match_for = set()
        websocket_mismatch = False

        def _match(state: State, parts: list[str], values: list[str]) -> tuple[Rule, list[str]] | None:
            nonlocal have_match_for, websocket_mismatch
            if parts == []:
                for rule in state.rules:
                    if rule.methods is not None and method not in rule.methods:
                        have_match_for.update(rule.methods)
                    elif rule.websocket != websocket:
                        websocket_mismatch = True
                    else:
                        return (rule, values)
                if '' in state.static:
                    for rule in state.static[''].rules:
                        if websocket == rule.websocket and (rule.methods is None or method in rule.methods):
                            if rule.strict_slashes:
                                raise SlashRequired()
                            else:
                                return (rule, values)
                return None
            part = parts[0]
            if part in state.static:
                rv = _match(state.static[part], parts[1:], values)
                if rv is not None:
                    return rv
            for test_part, new_state in state.dynamic:
                target = part
                remaining = parts[1:]
                if test_part.final:
                    target = '/'.join(parts)
                    remaining = []
                match = re.compile(test_part.content).match(target)
                if match is not None:
                    if test_part.suffixed:
                        suffix = match.groups()[-1]
                        if suffix == '/':
                            remaining = ['']
                    converter_groups = sorted(match.groupdict().items(), key=lambda entry: entry[0])
                    groups = [value for key, value in converter_groups if key[:11] == '__werkzeug_']
                    rv = _match(new_state, remaining, values + groups)
                    if rv is not None:
                        return rv
            if parts == ['']:
                for rule in state.rules:
                    if rule.strict_slashes:
                        continue
                    if rule.methods is not None and method not in rule.methods:
                        have_match_for.update(rule.methods)
                    elif rule.websocket != websocket:
                        websocket_mismatch = True
                    else:
                        return (rule, values)
            return None
        try:
            rv = _match(self._root, [domain, *path.split('/')], [])
        except SlashRequired:
            raise RequestPath(f'{path}/') from None
        if self.merge_slashes and rv is None:
            path = re.sub('/{2,}?', '/', path)
            try:
                rv = _match(self._root, [domain, *path.split('/')], [])
            except SlashRequired:
                raise RequestPath(f'{path}/') from None
            if rv is None:
                raise NoMatch(have_match_for, websocket_mismatch)
            else:
                raise RequestPath(f'{path}')
        elif rv is not None:
            rule, values = rv
            result = {}
            for name, value in zip(rule._converters.keys(), values):
                try:
                    value = rule._converters[name].to_python(value)
                except ValidationError:
                    raise NoMatch(have_match_for, websocket_mismatch) from None
                result[str(name)] = value
            if rule.defaults:
                result.update(rule.defaults)
            if rule.alias and rule.map.redirect_defaults:
                raise RequestAliasRedirect(result, rule.endpoint)
            return (rule, result)
        raise NoMatch(have_match_for, websocket_mismatch)