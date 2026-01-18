from collections import defaultdict
def _clear_string_cache(self):
    try:
        del self.__cached_str
    except AttributeError:
        pass