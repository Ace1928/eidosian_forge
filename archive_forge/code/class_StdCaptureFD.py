import os
import sys
import py
import tempfile
class StdCaptureFD(Capture):
    """ This class allows to capture writes to FD1 and FD2
        and may connect a NULL file to FD0 (and prevent
        reads from sys.stdin).  If any of the 0,1,2 file descriptors
        is invalid it will not be captured.
    """

    def __init__(self, out=True, err=True, mixed=False, in_=True, patchsys=True, now=True):
        self._options = {'out': out, 'err': err, 'mixed': mixed, 'in_': in_, 'patchsys': patchsys, 'now': now}
        self._save()
        if now:
            self.startall()

    def _save(self):
        in_ = self._options['in_']
        out = self._options['out']
        err = self._options['err']
        mixed = self._options['mixed']
        patchsys = self._options['patchsys']
        if in_:
            try:
                self.in_ = FDCapture(0, tmpfile=None, now=False, patchsys=patchsys)
            except OSError:
                pass
        if out:
            tmpfile = None
            if hasattr(out, 'write'):
                tmpfile = out
            try:
                self.out = FDCapture(1, tmpfile=tmpfile, now=False, patchsys=patchsys)
                self._options['out'] = self.out.tmpfile
            except OSError:
                pass
        if err:
            if out and mixed:
                tmpfile = self.out.tmpfile
            elif hasattr(err, 'write'):
                tmpfile = err
            else:
                tmpfile = None
            try:
                self.err = FDCapture(2, tmpfile=tmpfile, now=False, patchsys=patchsys)
                self._options['err'] = self.err.tmpfile
            except OSError:
                pass

    def startall(self):
        if hasattr(self, 'in_'):
            self.in_.start()
        if hasattr(self, 'out'):
            self.out.start()
        if hasattr(self, 'err'):
            self.err.start()

    def resume(self):
        """ resume capturing with original temp files. """
        self.startall()

    def done(self, save=True):
        """ return (outfile, errfile) and stop capturing. """
        outfile = errfile = None
        if hasattr(self, 'out') and (not self.out.tmpfile.closed):
            outfile = self.out.done()
        if hasattr(self, 'err') and (not self.err.tmpfile.closed):
            errfile = self.err.done()
        if hasattr(self, 'in_'):
            tmpfile = self.in_.done()
        if save:
            self._save()
        return (outfile, errfile)

    def readouterr(self):
        """ return snapshot value of stdout/stderr capturings. """
        if hasattr(self, 'out'):
            out = self._readsnapshot(self.out.tmpfile)
        else:
            out = ''
        if hasattr(self, 'err'):
            err = self._readsnapshot(self.err.tmpfile)
        else:
            err = ''
        return (out, err)

    def _readsnapshot(self, f):
        f.seek(0)
        res = f.read()
        enc = getattr(f, 'encoding', None)
        if enc:
            res = py.builtin._totext(res, enc, 'replace')
        f.truncate(0)
        f.seek(0)
        return res