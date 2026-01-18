from __future__ import absolute_import
import functools
import json
import re
import sys
def __lex_finish_number(self):
    s = self.buffer
    m = Parser.__number_re.match(s)
    if m:
        sign, integer, fraction, exp = m.groups()
        if exp is not None and (int(exp) > sys.maxsize or int(exp) < -sys.maxsize - 1):
            self.__error('exponent outside valid range')
            return
        if fraction is not None and len(fraction.lstrip('0')) == 0:
            fraction = None
        sig_string = integer
        if fraction is not None:
            sig_string += fraction
        significand = int(sig_string)
        pow10 = 0
        if fraction is not None:
            pow10 -= len(fraction)
        if exp is not None:
            pow10 += int(exp)
        if significand == 0:
            self.__parser_input(0)
            return
        elif significand <= 2 ** 63:
            while pow10 > 0 and significand <= 2 ** 63:
                significand *= 10
                pow10 -= 1
            while pow10 < 0 and significand % 10 == 0:
                significand //= 10
                pow10 += 1
            if pow10 == 0 and (not sign and significand < 2 ** 63 or (sign and significand <= 2 ** 63)):
                if sign:
                    self.__parser_input(-significand)
                else:
                    self.__parser_input(significand)
                return
        value = float(s)
        if value == float('inf') or value == float('-inf'):
            self.__error('number outside valid range')
            return
        if value == 0:
            value = 0
        self.__parser_input(value)
    elif re.match('-?0[0-9]', s):
        self.__error('leading zeros not allowed')
    elif re.match('-([^0-9]|$)', s):
        self.__error("'-' must be followed by digit")
    elif re.match('-?(0|[1-9][0-9]*)\\.([^0-9]|$)', s):
        self.__error('decimal point must be followed by digit')
    elif re.search('e[-+]?([^0-9]|$)', s):
        self.__error('exponent must contain at least one digit')
    else:
        self.__error('syntax error in number')