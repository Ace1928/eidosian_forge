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
class FzCookie(object):
    """
    Wrapper class for struct `fz_cookie`. Not copyable or assignable.
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

    def __init__(self):
        """ Default constructor sets all fields to default values."""
        _mupdf.FzCookie_swiginit(self, _mupdf.new_FzCookie())

    def set_abort(self):
        """ Sets m_internal.abort to 1."""
        return _mupdf.FzCookie_set_abort(self)

    def increment_errors(self, delta):
        """ Increments m_internal.errors by <delta>."""
        return _mupdf.FzCookie_increment_errors(self, delta)

    def abort(self):
        return _mupdf.FzCookie_abort(self)

    def progress(self):
        return _mupdf.FzCookie_progress(self)

    def progress_max(self):
        return _mupdf.FzCookie_progress_max(self)

    def errors(self):
        return _mupdf.FzCookie_errors(self)

    def incomplete(self):
        return _mupdf.FzCookie_incomplete(self)
    __swig_destroy__ = _mupdf.delete_FzCookie
    m_internal = property(_mupdf.FzCookie_m_internal_get, _mupdf.FzCookie_m_internal_set)
    s_num_instances = property(_mupdf.FzCookie_s_num_instances_get, _mupdf.FzCookie_s_num_instances_set, doc=' Wrapped data is held by value.')

    def to_string(self):
        """ Returns string containing our members, labelled and inside (...), using operator<<."""
        return _mupdf.FzCookie_to_string(self)

    def __eq__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzCookie___eq__(self, rhs)

    def __ne__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzCookie___ne__(self, rhs)