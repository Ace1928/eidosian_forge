import sys
import traceback
class _Truncated:

    def __init__(self):
        self.tb_lineno = -1
        self.tb_frame = _Object(f_globals={'__file__': '', '__name__': '', '__loader__': None}, f_fileno=None, f_code=_Object(co_filename='...', co_name='[rest of traceback truncated]'))
        self.tb_next = None
        self.tb_lasti = 0
    if sys.version_info >= (3, 11):

        @property
        def co_positions(self):
            return self.tb_frame.co_positions