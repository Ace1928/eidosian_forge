import gettext
import logging
import os
import warnings
import fixtures
from oslo_config import cfg
class TranslationFixture(fixtures.Fixture):
    """Use gettext NullTranslation objects in tests."""

    def setUp(self):
        super(TranslationFixture, self).setUp()
        nulltrans = gettext.NullTranslations()
        gettext_fixture = fixtures.MonkeyPatch('gettext.translation', lambda *x, **y: nulltrans)
        self.gettext_patcher = self.useFixture(gettext_fixture)