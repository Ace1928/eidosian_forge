import gettext
import fixtures
from oslo_i18n import _lazy
from oslo_i18n import _message
def _prefix_translations(*x, **y):
    """Use message id prefixed with domain and language as translation

    """
    return _PrefixTranslator(prefix=x[0] + '/' + y['languages'][0] + ': ')