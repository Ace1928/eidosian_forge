from . import errors, osutils
@staticmethod
def _step_one(iterator):
    """Step an iter_entries_by_dir iterator.

        :return: (has_more, path, ie)
            If has_more is False, path and ie will be None.
        """
    try:
        path, ie = next(iterator)
    except StopIteration:
        return (False, None, None)
    else:
        return (True, path, ie)