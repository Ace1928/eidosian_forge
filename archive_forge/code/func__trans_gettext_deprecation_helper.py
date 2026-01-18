import gettext
import os
import warnings
def _trans_gettext_deprecation_helper(*args, **kwargs):
    """The trans gettext deprecation helper."""
    warn_msg = 'The alias `_()` will be deprecated. Use `_i18n()` instead.'
    warnings.warn(warn_msg, FutureWarning, stacklevel=2)
    return trans.gettext(*args, **kwargs)