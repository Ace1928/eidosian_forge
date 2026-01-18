import re
from functools import partial
from tornado import httputil
from tornado.httpserver import _CallableAdapter
from tornado.escape import url_escape, url_unescape, utf8
from tornado.log import app_log
from tornado.util import basestring_type, import_object, re_unescape, unicode_type
from typing import Any, Union, Optional, Awaitable, List, Dict, Pattern, Tuple, overload
class PathMatches(Matcher):
    """Matches requests with paths specified by ``path_pattern`` regex."""

    def __init__(self, path_pattern: Union[str, Pattern]) -> None:
        if isinstance(path_pattern, basestring_type):
            if not path_pattern.endswith('$'):
                path_pattern += '$'
            self.regex = re.compile(path_pattern)
        else:
            self.regex = path_pattern
        assert len(self.regex.groupindex) in (0, self.regex.groups), 'groups in url regexes must either be all named or all positional: %r' % self.regex.pattern
        self._path, self._group_count = self._find_groups()

    def match(self, request: httputil.HTTPServerRequest) -> Optional[Dict[str, Any]]:
        match = self.regex.match(request.path)
        if match is None:
            return None
        if not self.regex.groups:
            return {}
        path_args = []
        path_kwargs = {}
        if self.regex.groupindex:
            path_kwargs = dict(((str(k), _unquote_or_none(v)) for k, v in match.groupdict().items()))
        else:
            path_args = [_unquote_or_none(s) for s in match.groups()]
        return dict(path_args=path_args, path_kwargs=path_kwargs)

    def reverse(self, *args: Any) -> Optional[str]:
        if self._path is None:
            raise ValueError('Cannot reverse url regex ' + self.regex.pattern)
        assert len(args) == self._group_count, 'required number of arguments not found'
        if not len(args):
            return self._path
        converted_args = []
        for a in args:
            if not isinstance(a, (unicode_type, bytes)):
                a = str(a)
            converted_args.append(url_escape(utf8(a), plus=False))
        return self._path % tuple(converted_args)

    def _find_groups(self) -> Tuple[Optional[str], Optional[int]]:
        """Returns a tuple (reverse string, group count) for a url.

        For example: Given the url pattern /([0-9]{4})/([a-z-]+)/, this method
        would return ('/%s/%s/', 2).
        """
        pattern = self.regex.pattern
        if pattern.startswith('^'):
            pattern = pattern[1:]
        if pattern.endswith('$'):
            pattern = pattern[:-1]
        if self.regex.groups != pattern.count('('):
            return (None, None)
        pieces = []
        for fragment in pattern.split('('):
            if ')' in fragment:
                paren_loc = fragment.index(')')
                if paren_loc >= 0:
                    try:
                        unescaped_fragment = re_unescape(fragment[paren_loc + 1:])
                    except ValueError:
                        return (None, None)
                    pieces.append('%s' + unescaped_fragment)
            else:
                try:
                    unescaped_fragment = re_unescape(fragment)
                except ValueError:
                    return (None, None)
                pieces.append(unescaped_fragment)
        return (''.join(pieces), self.regex.groups)