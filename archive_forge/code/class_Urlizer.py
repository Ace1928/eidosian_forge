import html
import json
import re
import warnings
from html.parser import HTMLParser
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.encoding import punycode
from django.utils.functional import Promise, keep_lazy, keep_lazy_text
from django.utils.http import RFC3986_GENDELIMS, RFC3986_SUBDELIMS
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import SafeData, SafeString, mark_safe
from django.utils.text import normalize_newlines
class Urlizer:
    """
    Convert any URLs in text into clickable links.

    Work on http://, https://, www. links, and also on links ending in one of
    the original seven gTLDs (.com, .edu, .gov, .int, .mil, .net, and .org).
    Links can have trailing punctuation (periods, commas, close-parens) and
    leading punctuation (opening parens) and it'll still do the right thing.
    """
    trailing_punctuation_chars = '.,:;!'
    wrapping_punctuation = [('(', ')'), ('[', ']')]
    simple_url_re = _lazy_re_compile('^https?://\\[?\\w', re.IGNORECASE)
    simple_url_2_re = _lazy_re_compile('^www\\.|^(?!http)\\w[^@]+\\.(com|edu|gov|int|mil|net|org)($|/.*)$', re.IGNORECASE)
    word_split_re = _lazy_re_compile('([\\s<>"\']+)')
    mailto_template = 'mailto:{local}@{domain}'
    url_template = '<a href="{href}"{attrs}>{url}</a>'

    def __call__(self, text, trim_url_limit=None, nofollow=False, autoescape=False):
        """
        If trim_url_limit is not None, truncate the URLs in the link text
        longer than this limit to trim_url_limit - 1 characters and append an
        ellipsis.

        If nofollow is True, give the links a rel="nofollow" attribute.

        If autoescape is True, autoescape the link text and URLs.
        """
        safe_input = isinstance(text, SafeData)
        words = self.word_split_re.split(str(text))
        return ''.join([self.handle_word(word, safe_input=safe_input, trim_url_limit=trim_url_limit, nofollow=nofollow, autoescape=autoescape) for word in words])

    def handle_word(self, word, *, safe_input, trim_url_limit=None, nofollow=False, autoescape=False):
        if '.' in word or '@' in word or ':' in word:
            lead, middle, trail = self.trim_punctuation(word)
            url = None
            nofollow_attr = ' rel="nofollow"' if nofollow else ''
            if self.simple_url_re.match(middle):
                url = smart_urlquote(html.unescape(middle))
            elif self.simple_url_2_re.match(middle):
                url = smart_urlquote('http://%s' % html.unescape(middle))
            elif ':' not in middle and self.is_email_simple(middle):
                local, domain = middle.rsplit('@', 1)
                try:
                    domain = punycode(domain)
                except UnicodeError:
                    return word
                url = self.mailto_template.format(local=local, domain=domain)
                nofollow_attr = ''
            if url:
                trimmed = self.trim_url(middle, limit=trim_url_limit)
                if autoescape and (not safe_input):
                    lead, trail = (escape(lead), escape(trail))
                    trimmed = escape(trimmed)
                middle = self.url_template.format(href=escape(url), attrs=nofollow_attr, url=trimmed)
                return mark_safe(f'{lead}{middle}{trail}')
            elif safe_input:
                return mark_safe(word)
            elif autoescape:
                return escape(word)
        elif safe_input:
            return mark_safe(word)
        elif autoescape:
            return escape(word)
        return word

    def trim_url(self, x, *, limit):
        if limit is None or len(x) <= limit:
            return x
        return '%s…' % x[:max(0, limit - 1)]

    def trim_punctuation(self, word):
        """
        Trim trailing and wrapping punctuation from `word`. Return the items of
        the new state.
        """
        lead, middle, trail = ('', word, '')
        trimmed_something = True
        while trimmed_something:
            trimmed_something = False
            for opening, closing in self.wrapping_punctuation:
                if middle.startswith(opening):
                    middle = middle.removeprefix(opening)
                    lead += opening
                    trimmed_something = True
                if middle.endswith(closing) and middle.count(closing) == middle.count(opening) + 1:
                    middle = middle.removesuffix(closing)
                    trail = closing + trail
                    trimmed_something = True
            middle_unescaped = html.unescape(middle)
            stripped = middle_unescaped.rstrip(self.trailing_punctuation_chars)
            if middle_unescaped != stripped:
                punctuation_count = len(middle_unescaped) - len(stripped)
                trail = middle[-punctuation_count:] + trail
                middle = middle[:-punctuation_count]
                trimmed_something = True
        return (lead, middle, trail)

    @staticmethod
    def is_email_simple(value):
        """Return True if value looks like an email address."""
        if '@' not in value or value.startswith('@') or value.endswith('@'):
            return False
        try:
            p1, p2 = value.split('@')
        except ValueError:
            return False
        if '.' not in p2 or p2.startswith('.'):
            return False
        return True