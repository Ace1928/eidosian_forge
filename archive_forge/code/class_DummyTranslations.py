from datetime import datetime
from gettext import NullTranslations
import unittest
import six
from genshi.core import Attrs
from genshi.template import MarkupTemplate, Context
from genshi.filters.i18n import Translator, extract
from genshi.input import HTML
from genshi.compat import IS_PYTHON2, StringIO
from genshi.tests.test_utils import doctest_suite
class DummyTranslations(NullTranslations):
    _domains = {}

    def __init__(self, catalog=()):
        NullTranslations.__init__(self)
        self._catalog = catalog or {}
        self.plural = lambda n: n != 1

    def add_domain(self, domain, catalog):
        translation = DummyTranslations(catalog)
        translation.add_fallback(self)
        self._domains[domain] = translation

    def _domain_call(self, func, domain, *args, **kwargs):
        return getattr(self._domains.get(domain, self), func)(*args, **kwargs)
    if IS_PYTHON2:

        def ugettext(self, message):
            missing = object()
            tmsg = self._catalog.get(message, missing)
            if tmsg is missing:
                if self._fallback:
                    return self._fallback.ugettext(message)
                return six.text_type(message)
            return tmsg
    else:

        def gettext(self, message):
            missing = object()
            tmsg = self._catalog.get(message, missing)
            if tmsg is missing:
                if self._fallback:
                    return self._fallback.gettext(message)
                return six.text_type(message)
            return tmsg
    if IS_PYTHON2:

        def dugettext(self, domain, message):
            return self._domain_call('ugettext', domain, message)
    else:

        def dgettext(self, domain, message):
            return self._domain_call('gettext', domain, message)

    def ungettext(self, msgid1, msgid2, n):
        try:
            return self._catalog[msgid1, self.plural(n)]
        except KeyError:
            if self._fallback:
                return self._fallback.ngettext(msgid1, msgid2, n)
            if n == 1:
                return msgid1
            else:
                return msgid2
    if not IS_PYTHON2:
        ngettext = ungettext
        del ungettext
    if IS_PYTHON2:

        def dungettext(self, domain, singular, plural, numeral):
            return self._domain_call('ungettext', domain, singular, plural, numeral)
    else:

        def dngettext(self, domain, singular, plural, numeral):
            return self._domain_call('ngettext', domain, singular, plural, numeral)
    if IS_PYTHON2:

        def upgettext(self, context, message):
            try:
                return self._catalog[context, message]
            except KeyError:
                if self._fallback:
                    return self._fallback.upgettext(context, message)
                return unicode(message)
    else:

        def pgettext(self, context, message):
            try:
                return self._catalog[context, message]
            except KeyError:
                if self._fallback:
                    return self._fallback.upgettext(context, message)
                return message
    if IS_PYTHON2:

        def dupgettext(self, domain, context, message):
            return self._domain_call('upgettext', domain, context, message)
    else:

        def dpgettext(self, domain, context, message):
            return self._domain_call('pgettext', domain, context, message)

    def unpgettext(self, context, msgid1, msgid2, n):
        try:
            return self._catalog[context, msgid1, self.plural(n)]
        except KeyError:
            if self._fallback:
                return self._fallback.unpgettext(context, msgid1, msgid2, n)
            if n == 1:
                return msgid1
            else:
                return msgid2
    if not IS_PYTHON2:
        npgettext = unpgettext
        del unpgettext
    if IS_PYTHON2:

        def dunpgettext(self, domain, context, msgid1, msgid2, n):
            return self._domain_call('unpgettext', context, msgid1, msgid2, n)
    else:

        def dnpgettext(self, domain, context, msgid1, msgid2, n):
            return self._domain_call('npgettext', context, msgid1, msgid2, n)