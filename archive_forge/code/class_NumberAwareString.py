from contextlib import ContextDecorator
from decimal import ROUND_UP, Decimal
from django.utils.autoreload import autoreload_started, file_changed
from django.utils.functional import lazy
from django.utils.regex_helper import _lazy_re_compile
class NumberAwareString(resultclass):

    def __bool__(self):
        return bool(kwargs['singular'])

    def _get_number_value(self, values):
        try:
            return values[number]
        except KeyError:
            raise KeyError("Your dictionary lacks key '%s'. Please provide it, because it is required to determine whether string is singular or plural." % number)

    def _translate(self, number_value):
        kwargs['number'] = number_value
        return func(**kwargs)

    def format(self, *args, **kwargs):
        number_value = self._get_number_value(kwargs) if kwargs and number else args[0]
        return self._translate(number_value).format(*args, **kwargs)

    def __mod__(self, rhs):
        if isinstance(rhs, dict) and number:
            number_value = self._get_number_value(rhs)
        else:
            number_value = rhs
        translated = self._translate(number_value)
        try:
            translated %= rhs
        except TypeError:
            pass
        return translated