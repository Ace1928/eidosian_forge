import os
import sys
import py
import tempfile
class StdCapture(Capture):
    """ This class allows to capture writes to sys.stdout|stderr "in-memory"
        and will raise errors on tries to read from sys.stdin. It only
        modifies sys.stdout|stderr|stdin attributes and does not
        touch underlying File Descriptors (use StdCaptureFD for that).
    """

    def __init__(self, out=True, err=True, in_=True, mixed=False, now=True):
        self._oldout = sys.stdout
        self._olderr = sys.stderr
        self._oldin = sys.stdin
        if out and (not hasattr(out, 'file')):
            out = TextIO()
        self.out = out
        if err:
            if mixed:
                err = out
            elif not hasattr(err, 'write'):
                err = TextIO()
        self.err = err
        self.in_ = in_
        if now:
            self.startall()

    def startall(self):
        if self.out:
            sys.stdout = self.out
        if self.err:
            sys.stderr = self.err
        if self.in_:
            sys.stdin = self.in_ = DontReadFromInput()

    def done(self, save=True):
        """ return (outfile, errfile) and stop capturing. """
        outfile = errfile = None
        if self.out and (not self.out.closed):
            sys.stdout = self._oldout
            outfile = self.out
            outfile.seek(0)
        if self.err and (not self.err.closed):
            sys.stderr = self._olderr
            errfile = self.err
            errfile.seek(0)
        if self.in_:
            sys.stdin = self._oldin
        return (outfile, errfile)

    def resume(self):
        """ resume capturing with original temp files. """
        self.startall()

    def readouterr(self):
        """ return snapshot value of stdout/stderr capturings. """
        out = err = ''
        if self.out:
            out = self.out.getvalue()
            self.out.truncate(0)
            self.out.seek(0)
        if self.err:
            err = self.err.getvalue()
            self.err.truncate(0)
            self.err.seek(0)
        return (out, err)