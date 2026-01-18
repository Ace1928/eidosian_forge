import os
import os.path
import re
from debian.deprecation import function_deprecated_by
import debian._arch_table
class NativeVersion(BaseVersion):
    """Represents a Debian package version, with native Python comparison"""
    re_all_digits_or_not = re.compile('\\d+|\\D+')
    re_digits = re.compile('\\d+')
    re_digit = re.compile('\\d')
    re_alpha = re.compile('[A-Za-z]')

    def _compare(self, other):
        if other is None:
            return 1
        if not isinstance(other, BaseVersion):
            try:
                other = BaseVersion(str(other))
            except ValueError as e:
                raise ValueError("Couldn't convert %r to BaseVersion: %s" % (other, e))
        lepoch = int(self.epoch or '0')
        repoch = int(other.epoch or '0')
        if lepoch < repoch:
            return -1
        if lepoch > repoch:
            return 1
        res = self._version_cmp_part(self.upstream_version or '0', other.upstream_version or '0')
        if res != 0:
            return res
        return self._version_cmp_part(self.debian_revision or '0', other.debian_revision or '0')

    @classmethod
    def _order(cls, x):
        """Return an integer value for character x"""
        if x == '~':
            return -1
        if cls.re_digit.match(x):
            return int(x) + 1
        if cls.re_alpha.match(x):
            return ord(x)
        return ord(x) + 256

    @classmethod
    def _version_cmp_string(cls, va, vb):
        la = [cls._order(x) for x in va]
        lb = [cls._order(x) for x in vb]
        while la or lb:
            a = 0
            b = 0
            if la:
                a = la.pop(0)
            if lb:
                b = lb.pop(0)
            if a < b:
                return -1
            if a > b:
                return 1
        return 0

    @classmethod
    def _version_cmp_part(cls, va, vb):
        la = cls.re_all_digits_or_not.findall(va)
        lb = cls.re_all_digits_or_not.findall(vb)
        while la or lb:
            a = '0'
            b = '0'
            if la:
                a = la.pop(0)
            if lb:
                b = lb.pop(0)
            if cls.re_digits.match(a) and cls.re_digits.match(b):
                aval = int(a)
                bval = int(b)
                if aval < bval:
                    return -1
                if aval > bval:
                    return 1
            else:
                res = cls._version_cmp_string(a, b)
                if res != 0:
                    return res
        return 0