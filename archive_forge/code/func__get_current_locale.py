import gettext as _gettext
import os
import sys
def _get_current_locale():
    if not os.environ.get('LANGUAGE'):
        from . import config
        lang = config.GlobalStack().get('language')
        if lang:
            os.environ['LANGUAGE'] = lang
            return lang
    if sys.platform == 'win32':
        _check_win32_locale()
    for i in ('LANGUAGE', 'LC_ALL', 'LC_MESSAGES', 'LANG'):
        lang = os.environ.get(i)
        if lang:
            return lang
    return None