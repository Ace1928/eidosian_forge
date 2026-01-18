import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
class CreditCardValidator(FormValidator):
    """
    Checks that credit card numbers are valid (if not real).

    You pass in the name of the field that has the credit card
    type and the field with the credit card number.  The credit
    card type should be one of "visa", "mastercard", "amex",
    "dinersclub", "discover", "jcb".

    You must check the expiration date yourself (there is no
    relation between CC number/types and expiration dates).

    ::

        >>> cc = CreditCardValidator()
        >>> sorted(cc.to_python({'ccType': 'visa', 'ccNumber': '4111111111111111'}).items())
        [('ccNumber', '4111111111111111'), ('ccType', 'visa')]
        >>> cc.to_python({'ccType': 'visa', 'ccNumber': '411111111111111'})
        Traceback (most recent call last):
            ...
        Invalid: ccNumber: You did not enter a valid number of digits
        >>> cc.to_python({'ccType': 'visa', 'ccNumber': '411111111111112'})
        Traceback (most recent call last):
            ...
        Invalid: ccNumber: You did not enter a valid number of digits
        >>> cc().to_python({})
        Traceback (most recent call last):
            ...
        Invalid: The field ccType is missing
    """
    validate_partial_form = True
    cc_type_field = 'ccType'
    cc_number_field = 'ccNumber'
    __unpackargs__ = ('cc_type_field', 'cc_number_field')
    messages = dict(notANumber=_('Please enter only the number, no other characters'), badLength=_('You did not enter a valid number of digits'), invalidNumber=_('That number is not valid'), missing_key=_('The field %(key)s is missing'))

    def validate_partial(self, field_dict, state):
        if not field_dict.get(self.cc_type_field, None) or not field_dict.get(self.cc_number_field, None):
            return None
        self._validate_python(field_dict, state)

    def _validate_python(self, field_dict, state):
        errors = self._validateReturn(field_dict, state)
        if errors:
            raise Invalid('<br>\n'.join(('%s: %s' % (name, value) for name, value in sorted(errors.items()))), field_dict, state, error_dict=errors)

    def _validateReturn(self, field_dict, state):
        for field in (self.cc_type_field, self.cc_number_field):
            if field not in field_dict:
                raise Invalid(self.message('missing_key', state, key=field), field_dict, state)
        ccType = field_dict[self.cc_type_field].lower().strip()
        number = field_dict[self.cc_number_field].strip()
        number = number.replace(' ', '')
        number = number.replace('-', '')
        try:
            int(number)
        except ValueError:
            return {self.cc_number_field: self.message('notANumber', state)}
        assert ccType in self._cardInfo, "I can't validate that type of credit card"
        foundValid = False
        validLength = False
        for prefix, length in self._cardInfo[ccType]:
            if len(number) == length:
                validLength = True
                if number.startswith(prefix):
                    foundValid = True
                    break
        if not validLength:
            return {self.cc_number_field: self.message('badLength', state)}
        if not foundValid:
            return {self.cc_number_field: self.message('invalidNumber', state)}
        if not self._validateMod10(number):
            return {self.cc_number_field: self.message('invalidNumber', state)}
        return None

    def _validateMod10(self, s):
        """Check string with the mod 10 algorithm (aka "Luhn formula")."""
        checksum, factor = (0, 1)
        for c in reversed(s):
            for c in str(factor * int(c)):
                checksum += int(c)
            factor = 3 - factor
        return checksum % 10 == 0
    _cardInfo = {'visa': [('4', 16), ('4', 13)], 'mastercard': [('51', 16), ('52', 16), ('53', 16), ('54', 16), ('55', 16)], 'discover': [('6011', 16)], 'amex': [('34', 15), ('37', 15)], 'dinersclub': [('300', 14), ('301', 14), ('302', 14), ('303', 14), ('304', 14), ('305', 14), ('36', 14), ('38', 14)], 'jcb': [('3', 16), ('2131', 15), ('1800', 15)]}