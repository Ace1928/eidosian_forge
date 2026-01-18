from collections import defaultdict
class _StringReferenceCaching(object):

    def get_reference(self):
        try:
            return self.__cached_refstr
        except AttributeError:
            s = self.__cached_refstr = self._get_reference()
            return s