import abc
import ast
import inspect
import stevedore
def _find_in_dict(self, test_value, path_segments, match):
    """Searches for a match in the dictionary.

        test_value is a reference inside the dictionary. Since the process is
        recursive, each call to _find_in_dict will be one level deeper.

        path_segments is the segments of the path to search.  The recursion
        ends when there are no more segments of path.

        When specifying a value inside a list, each element of the list is
        checked for a match. If the value is found within any of the sub lists
        the check succeeds; The check only fails if the entry is not in any of
        the sublists.

        """
    if len(path_segments) == 0:
        return match == str(test_value)
    key, path_segments = (path_segments[0], path_segments[1:])
    try:
        test_value = test_value[key]
    except KeyError:
        return False
    if isinstance(test_value, list):
        for val in test_value:
            if self._find_in_dict(val, path_segments, match):
                return True
        return False
    else:
        return self._find_in_dict(test_value, path_segments, match)