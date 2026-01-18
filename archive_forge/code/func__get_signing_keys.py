import os
from typing import Dict, List, Optional
from . import config, errors, trace, ui
from .i18n import gettext, ngettext
def _get_signing_keys(self):
    import gpg
    keyname = self._config_stack.get('gpg_signing_key')
    if keyname == 'default':
        return []
    if keyname:
        try:
            return [self.context.get_key(keyname)]
        except gpg.errors.KeyNotFound:
            pass
    if keyname is None:
        keyname = config.extract_email_address(self._config_stack.get('email'))
    if keyname == 'default':
        return []
    possible_keys = self.context.keylist(keyname, secret=True)
    try:
        return [next(possible_keys)]
    except StopIteration:
        return []