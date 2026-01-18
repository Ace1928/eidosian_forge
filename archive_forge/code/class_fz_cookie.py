from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
class fz_cookie(object):
    """
    Cookie support - simple communication channel between app/library.

    Provide two-way communication between application and library.
    Intended for multi-threaded applications where one thread is
    rendering pages and another thread wants to read progress
    feedback or abort a job that takes a long time to finish. The
    communication is unsynchronized without locking.

    abort: The application should set this field to 0 before
    calling fz_run_page to render a page. At any point when the
    page is being rendered the application my set this field to 1
    which will cause the rendering to finish soon. This field is
    checked periodically when the page is rendered, but exactly
    when is not known, therefore there is no upper bound on
    exactly when the rendering will abort. If the application
    did not provide a set of locks to fz_new_context, it must also
    await the completion of fz_run_page before issuing another
    call to fz_run_page. Note that once the application has set
    this field to 1 after it called fz_run_page it may not change
    the value again.

    progress: Communicates rendering progress back to the
    application and is read only. Increments as a page is being
    rendered. The value starts out at 0 and is limited to less
    than or equal to progress_max, unless progress_max is -1.

    progress_max: Communicates the known upper bound of rendering
    back to the application and is read only. The maximum value
    that the progress field may take. If there is no known upper
    bound on how long the rendering may take this value is -1 and
    progress is not limited. Note that the value of progress_max
    may change from -1 to a positive value once an upper bound is
    known, so take this into consideration when comparing the
    value of progress to that of progress_max.

    errors: count of errors during current rendering.

    incomplete: Initially should be set to 0. Will be set to
    non-zero if a TRYLATER error is thrown during rendering.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    abort = property(_mupdf.fz_cookie_abort_get, _mupdf.fz_cookie_abort_set)
    progress = property(_mupdf.fz_cookie_progress_get, _mupdf.fz_cookie_progress_set)
    progress_max = property(_mupdf.fz_cookie_progress_max_get, _mupdf.fz_cookie_progress_max_set)
    errors = property(_mupdf.fz_cookie_errors_get, _mupdf.fz_cookie_errors_set)
    incomplete = property(_mupdf.fz_cookie_incomplete_get, _mupdf.fz_cookie_incomplete_set)

    def __init__(self):
        _mupdf.fz_cookie_swiginit(self, _mupdf.new_fz_cookie())
    __swig_destroy__ = _mupdf.delete_fz_cookie