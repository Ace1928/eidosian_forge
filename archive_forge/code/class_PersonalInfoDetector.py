import importlib
import math
import re
from enum import Enum
class PersonalInfoDetector(object):
    """
    Detects whether a string contains any of the following personal information
    datapoints using regular expressions:

    - credit card
    - phone number
    - email
    - SSN
    """

    def __init__(self):
        self.credit_card_regex = '((?:(?:\\\\d{4}[- ]?){3}\\\\d{4}|\\\\d{15,16}))(?![\\\\d])'
        self.email_regex = "([a-z0-9!#$%&'*+\\/=?^_`{|.}~-]+@(?:[a-z0-9](?:[a-z0-9-]*" + '[a-z0-9])?\\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)'
        self.phone_number_regex = '\\D?(\\d{0,3}?)\\D{0,2}(\\d{3})?\\D{0,2}(\\d{3})\\D?(\\d{4})$'
        self.ssn_regex = '^\\d{3}-\\d{2}-\\d{4}$'

    def detect_all(self, text):
        contains = {}
        contains['credit_card'] = self.detect_credit_card(text)
        contains['email'] = self.detect_email(text)
        contains['phone_number'] = self.detect_phone_number(text)
        contains['ssn'] = self.detect_ssn(text)
        return contains

    def txt_format_detect_all(self, text):
        contains = self.detect_all(text)
        contains_personal_info = False
        txt = 'We believe this text contains the following personal ' + 'information:'
        for k, v in contains.items():
            if v != []:
                contains_personal_info = True
                txt += f'\n- {k.replace('_', ' ')}: {', '.join([str(x) for x in v])}'
        if not contains_personal_info:
            return ''
        return txt

    def detect_credit_card(self, text):
        return re.findall(self.credit_card_regex, text)

    def detect_email(self, text):
        text = text.lower()
        return re.findall(self.email_regex, text)

    def detect_phone_number(self, text):
        phones = re.findall(self.phone_number_regex, text)
        edited = []
        for tup in phones:
            edited.append(''.join(list(tup)))
        return edited

    def detect_ssn(self, text):
        return re.findall(self.ssn_regex, text)