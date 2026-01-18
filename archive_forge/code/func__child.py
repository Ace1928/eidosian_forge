import py
import os
import sys
import marshal
def _child(self, nice_level, child_on_start, child_on_exit):
    sys.stdout = stdout = get_unbuffered_io(1, self.STDOUT)
    sys.stderr = stderr = get_unbuffered_io(2, self.STDERR)
    retvalf = self.RETVAL.open('wb')
    EXITSTATUS = 0
    try:
        if nice_level:
            os.nice(nice_level)
        try:
            if child_on_start is not None:
                child_on_start()
            retval = self.fun(*self.args, **self.kwargs)
            retvalf.write(marshal.dumps(retval))
            if child_on_exit is not None:
                child_on_exit()
        except:
            excinfo = py.code.ExceptionInfo()
            stderr.write(str(excinfo._getreprcrash()))
            EXITSTATUS = self.EXITSTATUS_EXCEPTION
    finally:
        stdout.close()
        stderr.close()
        retvalf.close()
    os.close(1)
    os.close(2)
    os._exit(EXITSTATUS)