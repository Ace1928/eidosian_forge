import gettext
import fixtures
from oslo_i18n import _lazy
from oslo_i18n import _message
class _PrefixTranslator(gettext.NullTranslations):
    """Translator that adds prefix to message ids

    NOTE: gettext.NullTranslations is an old style class

    :parm prefix: prefix to add to message id.  If not specified (None)
                  then 'noprefix' is used.
    :type prefix: string

    """

    def __init__(self, fp=None, prefix='noprefix'):
        gettext.NullTranslations.__init__(self, fp)
        self.prefix = prefix

    def gettext(self, message):
        msg = gettext.NullTranslations.gettext(self, message)
        return self.prefix + msg

    def ugettext(self, message):
        msg = gettext.NullTranslations.ugettext(self, message)
        return self.prefix + msg