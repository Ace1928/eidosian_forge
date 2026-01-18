import doctest
import re
import decimal
class NumericOutputChecker(doctest.OutputChecker):
    """
    Implements doctest's OutputChecker, see documentation of
    NumericExample for examples.

    >>> N = NumericOutputChecker()

    >>> a   = "[3.499e-8, 4.5?e-8]"
    >>> b   = "[3.499e-8,   4.5?e-8]"

    >>> N.check_output(a, b, NUMERIC_DICT[12])
    True

    >>> b   = "[3.499999e-8,   3.2?e-8]"
    >>> N.check_output(a, b, NUMERIC_DICT[6])
    True
    >>> N.check_output(a, b, NUMERIC_DICT[9])
    False
    >>> N.formatted_compare_numeric(a, b, NUMERIC_DICT[9])
    'Numbers differed by 1.3E-8\\n\\nExpected     : 3.499e-8\\nGot          : 3.499999e-8\\nDifference          : 9.99E-12\\n\\nExpected     : 4.5?e-8\\nGot          : 3.2?e-8\\nDifference (FAILURE): 1.3E-8\\n'
    >>> N.compare_numeric(a, b, NUMERIC_DICT[12])
    ('NUMERIC', ([('3.499e-8', '3.499999e-8', True, Decimal('9.99E-12')), ('4.5?e-8', '3.2?e-8', True, Decimal('1.3E-8'))], Decimal('1.3E-8')))

    >>> b   = "[3.4999e-8,  4.5e-8]"
    >>> N.formatted_compare_numeric(a, b, NUMERIC_DICT[6])
    'Expected interval, but got 4.5e-8.'

    >>> b   = "[3.4999?e-8, 4.5e-8]"
    >>> N.formatted_compare_numeric(a, b, NUMERIC_DICT[6])
    'Expected number, but got 3.4999?e-8.'

    >>> b  = "a = [3.4999e-8,  4.5?e-8]"
    >>> N.formatted_compare_numeric(a, b, NUMERIC_DICT[6])
    'Text between numbers differs: Expected "[" but got "a = [" at position 0'

    >>> b  = "[3.4999e-8,  4.5?e-8, 5.63]"
    >>> N.formatted_compare_numeric(a, b, NUMERIC_DICT[6])
    'Expected 2 numbers but got 3 numbers.'

    >>> a   = "[4.5,       6.7e1,       2e+3]"
    >>> b   = "[4.5000001, 67.00000001, 2.0000000000000000001e+3]"
    >>> N.compare_numeric(a, b, NUMERIC_DICT[6])
    ('OK', None)
    >>> N.compare_numeric(a, b, NUMERIC_DICT[12])
    ('NUMERIC', ([('4.5', '4.5000001', True, Decimal('1E-7')), ('6.7e1', '67.00000001', True, Decimal('1E-8')), ('2e+3', '2.0000000000000000001e+3', False, Decimal('1E-16'))], Decimal('1E-7')))

    Account for pari adding a space before the E::

        >>> a   = "4.5e-9"
        >>> b   = "4.5 E-9"
        >>> N.compare_numeric(a, b, NUMERIC_DICT[12])
        ('OK', None)

    """

    def compare_numeric(self, want, got, optionflags):
        """
        Compares want and got by scanning for numbers. The numbers are
        compared using an epsilon extracted from optionflags. The text
        pieces between the numbers are compared falling back to the
        default implementation of OutputChecker.

        Returns a pair (status, data) where status is 'OK' if the
        comparison passed or indicates how it failed with data containing
        information that can be used to format the text explaining the
        differences.
        """
        split_want = re.split(number_re, want)
        split_got = re.split(number_re, got)
        if len(split_want) != len(split_got):
            return ('COUNT', (len(split_want) // number_split_stride, len(split_got) // number_split_stride))
        flags = optionflags | NUMERIC_DEFAULT_OPTIONFLAGS
        for i in range(0, len(split_want), number_split_stride):
            if not doctest.OutputChecker.check_output(self, split_want[i], split_got[i], flags):
                return ('TEXT', (split_want[i], split_got[i], i))
        epsilon = decimal.Decimal(0.1) ** get_precision(optionflags)
        rows = []
        max_diff = 0
        for i in range(1, len(split_want), number_split_stride):
            number_want = split_want[i]
            number_got = split_got[i]
            is_interval_want = bool(split_want[i + 2])
            is_interval_got = bool(split_got[i + 2])
            if is_interval_want != is_interval_got:
                return ('TYPE', (is_interval_want, number_got))
            decimal_want = to_decimal(split_want[i:i + number_group_count])
            decimal_got = to_decimal(split_got[i:i + number_group_count])
            diff = abs(decimal_want - decimal_got)
            failed = diff > epsilon
            max_diff = max(max_diff, diff)
            rows.append((number_want, number_got, failed, diff))
        if max_diff > epsilon:
            return ('NUMERIC', (rows, max_diff))
        return ('OK', None)

    def format_compare_numeric_result(self, status, data):
        """
        Formats a nice text from the result of compare_numeric.
        """
        if status == 'COUNT':
            return 'Expected %d numbers but got %d numbers.' % data
        elif status == 'TEXT':
            return 'Text between numbers differs: Expected "%s" but got "%s" at position %d' % data
        elif status == 'TYPE':
            is_interval_want, number_got = data
            if is_interval_want:
                k = 'interval'
            else:
                k = 'number'
            return 'Expected %s, but got %s.' % (k, number_got)
        elif status == 'NUMERIC':
            rows, max_diff = data
            result = 'Numbers differed by %s\n' % max_diff
            for number_want, number_got, failed, diff in rows:
                if result:
                    result += '\n'
                result += 'Expected     : %s\n' % number_want
                result += 'Got          : %s\n' % number_got
                if failed:
                    result += 'Difference (FAILURE): %s\n' % diff
                else:
                    result += 'Difference          : %s\n' % diff
            return result
        else:
            raise Exception('Internal error in OutputChecker.')

    def formatted_compare_numeric(self, want, got, optionflags):
        """
        Performs comparison of compare_numeric and returns formatted
        text.

        Only supposed to be used if comparison failed.
        """
        status, data = self.compare_numeric(want, got, optionflags)
        return self.format_compare_numeric_result(status, data)

    def check_output(self, want, got, optionflags):
        """
        Implementation of OutputChecker method.
        """
        if want == got:
            return True
        if optionflags & ALL_NUMERIC:
            status, data = self.compare_numeric(want, got, optionflags)
            return status == 'OK'
        else:
            return doctest.OutputChecker.check_output(self, want, got, optionflags)

    def output_difference(self, example, got, optionflags):
        """
        Implementation of OutputChecker method.
        """
        if not optionflags & ALL_NUMERIC or example.exc_msg:
            return doctest.OutputChecker.output_difference(self, example, got, optionflags)
        else:
            flags = optionflags | NUMERIC_DEFAULT_OPTIONFLAGS
            base_result = doctest.OutputChecker.output_difference(self, example, got, flags)
            compare_result = self.formatted_compare_numeric(example.want, got, optionflags)
            return base_result + '\nReason for failure: ' + compare_result + '\n'