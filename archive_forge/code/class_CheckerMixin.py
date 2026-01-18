from distutils.version import StrictVersion
import operator
class CheckerMixin(object):

    def check_version(self, *predicates, **kwargs):
        return compare(get_version(self), *predicates, **kwargs)

    def compare_version(self, *predicates, **kwargs):
        return compare(get_version(self), *predicates, exc=False, **kwargs)