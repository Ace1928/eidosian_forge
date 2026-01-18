import gettext
import os
from oslo_i18n import _lazy
from oslo_i18n import _locale
from oslo_i18n import _message
def _make_translation_func(self, domain=None):
    """Return a translation function ready for use with messages.

        The returned function takes a single value, the unicode string
        to be translated.  The return type varies depending on whether
        lazy translation is being done. When lazy translation is
        enabled, :class:`Message` objects are returned instead of
        regular :class:`unicode` strings.

        The domain argument can be specified to override the default
        from the factory, but the localedir from the factory is always
        used because we assume the log-level translation catalogs are
        installed in the same directory as the main application
        catalog.

        """
    if domain is None:
        domain = self.domain
    t = gettext.translation(domain, localedir=self.localedir, fallback=True)
    m = t.gettext

    def f(msg):
        """oslo_i18n.gettextutils translation function."""
        if _lazy.USE_LAZY:
            return _message.Message(msg, domain=domain)
        return m(msg)
    return f