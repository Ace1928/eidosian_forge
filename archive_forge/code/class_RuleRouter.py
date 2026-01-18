import re
from functools import partial
from tornado import httputil
from tornado.httpserver import _CallableAdapter
from tornado.escape import url_escape, url_unescape, utf8
from tornado.log import app_log
from tornado.util import basestring_type, import_object, re_unescape, unicode_type
from typing import Any, Union, Optional, Awaitable, List, Dict, Pattern, Tuple, overload
class RuleRouter(Router):
    """Rule-based router implementation."""

    def __init__(self, rules: Optional[_RuleList]=None) -> None:
        """Constructs a router from an ordered list of rules::

            RuleRouter([
                Rule(PathMatches("/handler"), Target),
                # ... more rules
            ])

        You can also omit explicit `Rule` constructor and use tuples of arguments::

            RuleRouter([
                (PathMatches("/handler"), Target),
            ])

        `PathMatches` is a default matcher, so the example above can be simplified::

            RuleRouter([
                ("/handler", Target),
            ])

        In the examples above, ``Target`` can be a nested `Router` instance, an instance of
        `~.httputil.HTTPServerConnectionDelegate` or an old-style callable,
        accepting a request argument.

        :arg rules: a list of `Rule` instances or tuples of `Rule`
            constructor arguments.
        """
        self.rules = []
        if rules:
            self.add_rules(rules)

    def add_rules(self, rules: _RuleList) -> None:
        """Appends new rules to the router.

        :arg rules: a list of Rule instances (or tuples of arguments, which are
            passed to Rule constructor).
        """
        for rule in rules:
            if isinstance(rule, (tuple, list)):
                assert len(rule) in (2, 3, 4)
                if isinstance(rule[0], basestring_type):
                    rule = Rule(PathMatches(rule[0]), *rule[1:])
                else:
                    rule = Rule(*rule)
            self.rules.append(self.process_rule(rule))

    def process_rule(self, rule: 'Rule') -> 'Rule':
        """Override this method for additional preprocessing of each rule.

        :arg Rule rule: a rule to be processed.
        :returns: the same or modified Rule instance.
        """
        return rule

    def find_handler(self, request: httputil.HTTPServerRequest, **kwargs: Any) -> Optional[httputil.HTTPMessageDelegate]:
        for rule in self.rules:
            target_params = rule.matcher.match(request)
            if target_params is not None:
                if rule.target_kwargs:
                    target_params['target_kwargs'] = rule.target_kwargs
                delegate = self.get_target_delegate(rule.target, request, **target_params)
                if delegate is not None:
                    return delegate
        return None

    def get_target_delegate(self, target: Any, request: httputil.HTTPServerRequest, **target_params: Any) -> Optional[httputil.HTTPMessageDelegate]:
        """Returns an instance of `~.httputil.HTTPMessageDelegate` for a
        Rule's target. This method is called by `~.find_handler` and can be
        extended to provide additional target types.

        :arg target: a Rule's target.
        :arg httputil.HTTPServerRequest request: current request.
        :arg target_params: additional parameters that can be useful
            for `~.httputil.HTTPMessageDelegate` creation.
        """
        if isinstance(target, Router):
            return target.find_handler(request, **target_params)
        elif isinstance(target, httputil.HTTPServerConnectionDelegate):
            assert request.connection is not None
            return target.start_request(request.server_connection, request.connection)
        elif callable(target):
            assert request.connection is not None
            return _CallableAdapter(partial(target, **target_params), request.connection)
        return None