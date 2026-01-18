import gettext
class FakeTranslations(gettext.GNUTranslations):
    """A test GNUTranslations class that takes a map of msg -> translations."""

    def __init__(self, translations):
        self.translations = translations

    def gettext(self, msgid):
        return self.translations.get(msgid, msgid)

    def ugettext(self, msgid):
        return self.translations.get(msgid, msgid)

    @staticmethod
    def translator(locales_map):
        """Build mock translator for the given locales.

        Returns a mock gettext.translation function that uses
        individual TestTranslations to translate in the given locales.

        :param locales_map: A map from locale name to a translations map.
                            {
                             'es': {'Hi': 'Hola', 'Bye': 'Adios'},
                             'zh': {'Hi': 'Ni Hao', 'Bye': 'Zaijian'}
                            }


        """

        def _translation(domain, localedir=None, languages=None, fallback=None):
            if languages:
                language = languages[0]
                if language in locales_map:
                    return FakeTranslations(locales_map[language])
            return gettext.NullTranslations()
        return _translation