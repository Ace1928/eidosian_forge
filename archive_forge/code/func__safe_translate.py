import copy
import gettext
import locale
import logging
import os
import warnings
from oslo_i18n import _locale
from oslo_i18n import _translate
def _safe_translate(self, translated_message, translated_params):
    """Trap translation errors and fall back to default translation.

        :param translated_message: the requested translation

        :param translated_params: the params to be inserted

        :return: if parameter insertion is successful then it is the
                 translated_message with the translated_params inserted, if the
                 requested translation fails then it is the default translation
                 with the params
        """
    try:
        translated_message = translated_message % translated_params
    except (KeyError, TypeError) as err:
        msg = 'Failed to insert replacement values into translated message %s (Original: %r): %s'
        warnings.warn(msg % (translated_message, self.msgid, err))
        LOG.debug(msg, translated_message, self.msgid, err)
        translated_message = self.msgid % translated_params
    return translated_message