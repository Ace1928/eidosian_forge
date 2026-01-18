import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def _handle_err_list(ret, errlist, names, exception, mapper):
    """
    Convert one or more errors from an operation into the requested exception.

    :param int ret: the overall return code.
    :param errlist: the dictionary that maps entity names to their specific error codes.
    :type errlist: dict of bytes:int
    :param names: the list of all names of the entities on which the operation was attempted.
    :param type exception: the type of the exception to raise if an error occurred.
                           The exception should be a subclass of `MultipleOperationsFailure`.
    :param function mapper: the function that maps an error code and a name to a Python exception.

    Unless ``ret`` is zero this function will raise the ``exception``.
    If the ``errlist`` is not empty, then the compound exception will contain a list of exceptions
    corresponding to each individual error code in the ``errlist``.
    Otherwise, the ``exception`` will contain a list with a single exception corresponding to the
    ``ret`` value.  If the ``names`` list contains only one element, that is, the operation was
    attempted on a single entity, then the name of that entity is passed to the ``mapper``.
    If the operation was attempted on multiple entities, but the ``errlist`` is empty, then we
    can not know which entity caused the error and, thus, ``None`` is used as a name to signify
    thati fact.

    .. note::
        Note that the ``errlist`` can contain a special element with a key of "N_MORE_ERRORS".
        That element means that there were too many errors to place on the ``errlist``.
        Those errors are suppressed and only their count is provided as a value of the special
        ``N_MORE_ERRORS`` element.
    """
    if ret == 0:
        return
    if len(errlist) == 0:
        suppressed_count = 0
        if len(names) == 1:
            name = names[0]
        else:
            name = None
        errors = [mapper(ret, name)]
    else:
        errors = []
        suppressed_count = errlist.pop('N_MORE_ERRORS', 0)
        for name, err in errlist.iteritems():
            errors.append(mapper(err, name))
    raise exception(errors, suppressed_count)