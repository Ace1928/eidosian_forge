import gettext
import fixtures
from oslo_i18n import _lazy
from oslo_i18n import _message
class PrefixLazyTranslation(fixtures.Fixture):
    """Fixture to prefix lazy translation enabled messages

    Use of this fixture will cause messages supporting lazy translation to
    be replaced with the message id prefixed with 'domain/language:'.
    For example, 'oslo/en_US: message about something'.  It will also
    override the available languages returned from
    oslo_18n.get_available_languages to the specified languages.

    This will enable tests to ensure that messages were translated lazily
    with the specified language and not immediately with the default language.

    NOTE that this does not work unless lazy translation is enabled, so it
    uses the ToggleLazy fixture to enable lazy translation.

    :param languages: list of languages to support.  If not specified (None)
                      then ['en_US'] is used.
    :type languages: list of strings

    """
    _DEFAULT_LANG = 'en_US'

    def __init__(self, languages=None, locale=None):
        super(PrefixLazyTranslation, self).__init__()
        self.languages = languages or [PrefixLazyTranslation._DEFAULT_LANG]
        self.locale = locale

    def setUp(self):
        super(PrefixLazyTranslation, self).setUp()
        self.useFixture(ToggleLazy(True))
        self.useFixture(fixtures.MonkeyPatch('oslo_i18n._gettextutils.get_available_languages', lambda *x, **y: self.languages))
        self.useFixture(fixtures.MonkeyPatch('oslo_i18n.get_available_languages', lambda *x, **y: self.languages))
        self.useFixture(fixtures.MonkeyPatch('gettext.translation', _prefix_translations))
        self.useFixture(fixtures.MonkeyPatch('locale.getlocale', lambda *x, **y: self.locale))