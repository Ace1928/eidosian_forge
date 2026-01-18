import gettext
import os
from oslo_i18n import _lazy
from oslo_i18n import _locale
from oslo_i18n import _message
def _make_plural_translation_func(self, domain=None):
    """Return a plural translation function ready for use with messages.

        The returned function takes three values, the single form of
        the unicode string, the plural form of the unicode string,
        the count of items to be translated.
        The returned type is the same as
        :method:`TranslatorFactory._make_translation_func`.

        The domain argument is the same as
        :method:`TranslatorFactory._make_translation_func`.

        """
    if domain is None:
        domain = self.domain
    t = gettext.translation(domain, localedir=self.localedir, fallback=True)
    m = t.ngettext

    def f(msgsingle, msgplural, msgcount):
        """oslo.i18n.gettextutils plural translation function."""
        if _lazy.USE_LAZY:
            msgid = (msgsingle, msgplural, msgcount)
            return _message.Message(msgid, domain=domain, has_plural_form=True)
        return m(msgsingle, msgplural, msgcount)
    return f