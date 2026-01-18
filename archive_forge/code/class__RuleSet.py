import logging
import re
from collections import namedtuple
from datetime import time
from urllib.parse import ParseResult, quote, urlparse, urlunparse
class _RuleSet(object):
    """Internal class which stores rules for a user agent."""

    def __init__(self, parser_instance):
        self.user_agent = None
        self._rules = []
        self._crawl_delay = None
        self._req_rate = None
        self._visit_time = None
        self._parser_instance = parser_instance

    def applies_to(self, robotname):
        """Return matching score."""
        robotname = robotname.strip().lower()
        if self.user_agent == '*':
            return 1
        if self.user_agent in robotname:
            return len(self.user_agent)
        return 0

    def _unquote(self, url, ignore='', errors='replace'):
        """Replace %xy escapes by their single-character equivalent."""
        if '%' not in url:
            return url

        def hex_to_byte(h):
            """Replaces a %xx escape with equivalent binary sequence."""
            return bytes.fromhex(h)
        ignore = {'{ord_c:02X}'.format(ord_c=ord(c)) for c in ignore}
        parts = url.split('%')
        parts[0] = parts[0].encode('utf-8')
        for i in range(1, len(parts)):
            if len(parts[i]) >= 2:
                if set(parts[i][:2]).issubset(_HEX_DIGITS):
                    hexcode = parts[i][:2].upper()
                    leftover = parts[i][2:]
                    if hexcode not in ignore:
                        parts[i] = hex_to_byte(hexcode) + leftover.encode('utf-8')
                        continue
                    else:
                        parts[i] = hexcode + leftover
            parts[i] = b'%' + parts[i].encode('utf-8')
        return b''.join(parts).decode('utf-8', errors)

    def hexescape(self, char):
        """Escape char as RFC 2396 specifies"""
        hex_repr = hex(ord(char))[2:].upper()
        if len(hex_repr) == 1:
            hex_repr = '0%s' % hex_repr
        return '%' + hex_repr

    def _quote_path(self, path):
        """Return percent encoded path."""
        parts = urlparse(path)
        path = self._unquote(parts.path, ignore='/%')
        path = quote(path, safe='/%')
        parts = ParseResult('', '', path, parts.params, parts.query, parts.fragment)
        path = urlunparse(parts)
        return path or '/'

    def _quote_pattern(self, pattern):
        if pattern.startswith('https://') or pattern.startswith('http://'):
            pattern = '/' + pattern
        last_char = ''
        if pattern[-1] == '?' or pattern[-1] == ';' or pattern[-1] == '$':
            last_char = pattern[-1]
            pattern = pattern[:-1]
        parts = urlparse(pattern)
        pattern = self._unquote(parts.path, ignore='/*$%')
        pattern = quote(pattern, safe='/*%=')
        parts = ParseResult('', '', pattern + last_char, parts.params, parts.query, parts.fragment)
        pattern = urlunparse(parts)
        return pattern

    def allow(self, pattern):
        if '$' in pattern:
            self.allow(pattern.replace('$', self.hexescape('$')))
        pattern = self._quote_pattern(pattern)
        if not pattern:
            return
        self._rules.append(_Rule(field='allow', value=_URLPattern(pattern)))
        if pattern.endswith('/index.html'):
            self.allow(pattern[:-10] + '$')

    def disallow(self, pattern):
        if '$' in pattern:
            self.disallow(pattern.replace('$', self.hexescape('$')))
        pattern = self._quote_pattern(pattern)
        if not pattern:
            return
        self._rules.append(_Rule(field='disallow', value=_URLPattern(pattern)))

    def finalize_rules(self):
        self._rules.sort(key=lambda r: (r.value.priority, r.field == 'allow'), reverse=True)

    def can_fetch(self, url):
        """Return if the url can be fetched."""
        url = self._quote_path(url)
        allowed = True
        for rule in self._rules:
            if rule.value.match(url):
                if rule.field == 'disallow':
                    allowed = False
                break
        return allowed

    @property
    def crawl_delay(self):
        """Get & set crawl delay for the rule set."""
        return self._crawl_delay

    @crawl_delay.setter
    def crawl_delay(self, delay):
        try:
            delay = float(delay)
        except ValueError:
            logger.debug("Malformed rule at line {line_seen} : cannot set crawl delay to '{delay}'. Ignoring this rule.".format(line_seen=self._parser_instance._total_line_seen, delay=delay))
            return
        self._crawl_delay = delay

    @property
    def request_rate(self):
        """Get & set request rate for the rule set."""
        return self._req_rate

    @request_rate.setter
    def request_rate(self, value):
        try:
            parts = value.split()
            if len(parts) == 2:
                rate, time_period = parts
            else:
                rate, time_period = (parts[0], '')
            requests, seconds = rate.split('/')
            time_unit = seconds[-1].lower()
            requests, seconds = (int(requests), int(seconds[:-1]))
            if time_unit == 'm':
                seconds *= 60
            elif time_unit == 'h':
                seconds *= 3600
            elif time_unit == 'd':
                seconds *= 86400
            start_time = None
            end_time = None
            if time_period:
                start_time, end_time = self._parse_time_period(time_period)
        except Exception:
            logger.debug("Malformed rule at line {line_seen} : cannot set request rate using '{value}'. Ignoring this rule.".format(line_seen=self._parser_instance._total_line_seen, value=value))
            return
        self._req_rate = RequestRate(requests, seconds, start_time, end_time)

    def _parse_time_period(self, time_period, separator='-'):
        """Parse a string with a time period into a tuple of start and end times."""
        start_time, end_time = time_period.split(separator)
        start_time = time(int(start_time[:2]), int(start_time[-2:]))
        end_time = time(int(end_time[:2]), int(end_time[-2:]))
        return (start_time, end_time)

    @property
    def visit_time(self):
        """Get & set visit time for the rule set."""
        return self._visit_time

    @visit_time.setter
    def visit_time(self, value):
        try:
            start_time, end_time = self._parse_time_period(value, separator=' ')
        except Exception:
            logger.debug("Malformed rule at line {line_seen} : cannot set visit time using '{value}'. Ignoring this rule.".format(line_seen=self._parser_instance._total_line_seen, value=value))
            return
        self._visit_time = VisitTime(start_time, end_time)