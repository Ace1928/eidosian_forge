import pprint
import re
import typing as t
from markupsafe import Markup
from . import defaults
from . import nodes
from .environment import Environment
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .runtime import concat  # type: ignore
from .runtime import Context
from .runtime import Undefined
from .utils import import_string
from .utils import pass_context
def _install_null(self, newstyle: t.Optional[bool]=None) -> None:
    import gettext
    translations = gettext.NullTranslations()
    if hasattr(translations, 'pgettext'):
        pgettext = translations.pgettext
    else:

        def pgettext(c: str, s: str) -> str:
            return s
    if hasattr(translations, 'npgettext'):
        npgettext = translations.npgettext
    else:

        def npgettext(c: str, s: str, p: str, n: int) -> str:
            return s if n == 1 else p
    self._install_callables(gettext=translations.gettext, ngettext=translations.ngettext, newstyle=newstyle, pgettext=pgettext, npgettext=npgettext)