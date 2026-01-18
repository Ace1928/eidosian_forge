import io
from .. import errors, i18n, tests, workingtree
class ZzzTranslations:
    """Special Zzz translation for debugging i18n stuff.

    This class can be used to confirm that the message is properly translated
    during black box tests.
    """
    _null_translation = i18n._gettext.NullTranslations()

    def zzz(self, s):
        return 'zzå{{%s}}' % s

    def gettext(self, s):
        return self.zzz(self._null_translation.gettext(s))

    def ngettext(self, s, p, n):
        return self.zzz(self._null_translation.ngettext(s, p, n))

    def ugettext(self, s):
        return self.zzz(self._null_translation.ugettext(s))

    def ungettext(self, s, p, n):
        return self.zzz(self._null_translation.ungettext(s, p, n))