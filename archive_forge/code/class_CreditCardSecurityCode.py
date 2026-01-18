import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
class CreditCardSecurityCode(FormValidator):
    """
    Checks that credit card security code has the correct number
    of digits for the given credit card type.

    You pass in the name of the field that has the credit card
    type and the field with the credit card security code.

    ::

        >>> code = CreditCardSecurityCode()
        >>> sorted(code.to_python({'ccType': 'visa', 'ccCode': '111'}).items())
        [('ccCode', '111'), ('ccType', 'visa')]
        >>> code.to_python({'ccType': 'visa', 'ccCode': '1111'})
        Traceback (most recent call last):
            ...
        Invalid: ccCode: Invalid credit card security code length
    """
    validate_partial_form = True
    cc_type_field = 'ccType'
    cc_code_field = 'ccCode'
    __unpackargs__ = ('cc_type_field', 'cc_code_field')
    messages = dict(notANumber=_('Please enter numbers only for credit card security code'), badLength=_('Invalid credit card security code length'))

    def validate_partial(self, field_dict, state):
        if not field_dict.get(self.cc_type_field, None) or not field_dict.get(self.cc_code_field, None):
            return None
        self._validate_python(field_dict, state)

    def _validate_python(self, field_dict, state):
        errors = self._validateReturn(field_dict, state)
        if errors:
            raise Invalid('<br>\n'.join(('%s: %s' % (name, value) for name, value in errors.items())), field_dict, state, error_dict=errors)

    def _validateReturn(self, field_dict, state):
        ccType = str(field_dict[self.cc_type_field]).strip()
        ccCode = str(field_dict[self.cc_code_field]).strip()
        try:
            int(ccCode)
        except ValueError:
            return {self.cc_code_field: self.message('notANumber', state)}
        length = self._cardInfo[ccType]
        if len(ccCode) != length:
            return {self.cc_code_field: self.message('badLength', state)}
    _cardInfo = dict(visa=3, mastercard=3, discover=3, amex=4)