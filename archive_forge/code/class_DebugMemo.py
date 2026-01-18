import os, pickle, sys, time, types, datetime, importlib
from ast import literal_eval
from base64 import decodebytes as base64_decodebytes, encodebytes as base64_encodebytes
from io import BytesIO
from hashlib import md5
from reportlab.lib.rltempfile import get_rl_tempfile, get_rl_tempdir
from . rl_safe_eval import rl_safe_exec, rl_safe_eval, safer_globals, rl_extended_literal_eval
from PIL import Image
import builtins
import reportlab
import glob, fnmatch
from urllib.parse import unquote, urlparse
from urllib.request import urlopen
from importlib import util as importlib_util
import itertools
class DebugMemo:
    """Intended as a simple report back encapsulator

    Typical usages:
        
    1. To record error data::
        
        dbg = DebugMemo(fn='dbgmemo.dbg',myVar=value)
        dbg.add(anotherPayload='aaaa',andagain='bbb')
        dbg.dump()

    2. To show the recorded info::
        
        dbg = DebugMemo(fn='dbgmemo.dbg',mode='r')
        dbg.load()
        dbg.show()

    3. To re-use recorded information::
        
        dbg = DebugMemo(fn='dbgmemo.dbg',mode='r')
            dbg.load()
        myTestFunc(dbg.payload('myVar'),dbg.payload('andagain'))

    In addition to the payload variables the dump records many useful bits
    of information which are also printed in the show() method.
    """

    def __init__(self, fn='rl_dbgmemo.dbg', mode='w', getScript=1, modules=(), capture_traceback=1, stdout=None, **kw):
        import socket
        self.fn = fn
        if not stdout:
            self.stdout = sys.stdout
        elif hasattr(stdout, 'write'):
            self.stdout = stdout
        else:
            self.stdout = open(stdout, 'w')
        if mode != 'w':
            return
        self.store = store = {}
        if capture_traceback and sys.exc_info() != (None, None, None):
            import traceback
            s = BytesIO()
            traceback.print_exc(None, s)
            store['__traceback'] = s.getvalue()
        cwd = os.getcwd()
        lcwd = os.listdir(cwd)
        pcwd = os.path.dirname(cwd)
        lpcwd = pcwd and os.listdir(pcwd) or '???'
        exed = os.path.abspath(os.path.dirname(sys.argv[0]))
        project_version = '???'
        md = None
        try:
            import marshal
            md = marshal.loads(__rl_loader__.get_data('meta_data.mar'))
            project_version = md['project_version']
        except:
            pass
        env = os.environ
        K = list(env.keys())
        K.sort()
        store.update({'gmt': time.asctime(time.gmtime(time.time())), 'platform': sys.platform, 'version': sys.version, 'hexversion': hex(sys.hexversion), 'executable': sys.executable, 'exec_prefix': sys.exec_prefix, 'prefix': sys.prefix, 'path': sys.path, 'argv': sys.argv, 'cwd': cwd, 'hostname': socket.gethostname(), 'lcwd': lcwd, 'lpcwd': lpcwd, 'byteorder': sys.byteorder, 'maxint': getattr(sys, 'maxunicode', '????'), 'api_version': getattr(sys, 'api_version', '????'), 'version_info': getattr(sys, 'version_info', '????'), 'winver': getattr(sys, 'winver', '????'), 'environment': '\n\t\t\t'.join([''] + ['%s=%r' % (k, env[k]) for k in K]), '__rl_loader__': repr(__rl_loader__), 'project_meta_data': md, 'project_version': project_version})
        for M, A in ((sys, ('getwindowsversion', 'getfilesystemencoding')), (os, ('uname', 'ctermid', 'getgid', 'getuid', 'getegid', 'geteuid', 'getlogin', 'getgroups', 'getpgrp', 'getpid', 'getppid'))):
            for a in A:
                if hasattr(M, a):
                    try:
                        store[a] = getattr(M, a)()
                    except:
                        pass
        if exed != cwd:
            try:
                store.update({'exed': exed, 'lexed': os.listdir(exed)})
            except:
                pass
        if getScript:
            fn = os.path.abspath(sys.argv[0])
            if os.path.isfile(fn):
                try:
                    store['__script'] = (fn, open(fn, 'r').read())
                except:
                    pass
        module_versions = {}
        for n, m in sys.modules.items():
            if n == 'reportlab' or n == 'rlextra' or n[:10] == 'reportlab.' or (n[:8] == 'rlextra.'):
                v = [getattr(m, x, None) for x in ('__version__', '__path__', '__file__')]
                if [_f for _f in v if _f]:
                    v = [v[0]] + [_f for _f in v[1:] if _f]
                    module_versions[n] = tuple(v)
        store['__module_versions'] = module_versions
        self.store['__payload'] = {}
        self._add(kw)

    def _add(self, D):
        payload = self.store['__payload']
        for k, v in D.items():
            payload[k] = v

    def add(self, **kw):
        self._add(kw)

    def _dump(self, f):
        try:
            pos = f.tell()
            pickle.dump(self.store, f)
        except:
            S = self.store.copy()
            ff = BytesIO()
            for k, v in S.items():
                try:
                    pickle.dump({k: v}, ff)
                except:
                    S[k] = '<unpicklable object %r>' % v
            f.seek(pos, 0)
            pickle.dump(S, f)

    def dump(self):
        f = open(self.fn, 'wb')
        try:
            self._dump(f)
        finally:
            f.close()

    def dumps(self):
        f = BytesIO()
        self._dump(f)
        return f.getvalue()

    def _load(self, f):
        self.store = pickle.load(f)

    def load(self):
        f = open(self.fn, 'rb')
        try:
            self._load(f)
        finally:
            f.close()

    def loads(self, s):
        self._load(BytesIO(s))

    def _show_module_versions(self, k, v):
        self._writeln(k[2:])
        K = list(v.keys())
        K.sort()
        for k in K:
            vk = vk0 = v[k]
            if isinstance(vk, tuple):
                vk0 = vk[0]
            try:
                __import__(k)
                m = sys.modules[k]
                d = getattr(m, '__version__', None) == vk0 and 'SAME' or 'DIFFERENT'
            except:
                m = None
                d = '??????unknown??????'
            self._writeln('  %s = %s (%s)' % (k, vk, d))

    def _banner(self, k, what):
        self._writeln('###################%s %s##################' % (what, k[2:]))

    def _start(self, k):
        self._banner(k, 'Start  ')

    def _finish(self, k):
        self._banner(k, 'Finish ')

    def _show_lines(self, k, v):
        self._start(k)
        self._writeln(v)
        self._finish(k)

    def _show_file(self, k, v):
        k = '%s %s' % (k, os.path.basename(v[0]))
        self._show_lines(k, v[1])

    def _show_payload(self, k, v):
        if v:
            import pprint
            self._start(k)
            pprint.pprint(v, self.stdout)
            self._finish(k)

    def _show_extensions(self):
        for mn in ('_rl_accel', '_renderPM', 'sgmlop', 'pyRXP', 'pyRXPU', '_imaging', 'Image'):
            try:
                A = [mn].append
                __import__(mn)
                m = sys.modules[mn]
                A(m.__file__)
                for vn in ('__version__', 'VERSION', '_version', 'version'):
                    if hasattr(m, vn):
                        A('%s=%r' % (vn, getattr(m, vn)))
            except:
                A('not found')
            self._writeln(' ' + ' '.join(A.__self__))
    specials = {'__module_versions': _show_module_versions, '__payload': _show_payload, '__traceback': _show_lines, '__script': _show_file}

    def show(self):
        K = list(self.store.keys())
        K.sort()
        for k in K:
            if k not in list(self.specials.keys()):
                self._writeln('%-15s = %s' % (k, self.store[k]))
        for k in K:
            if k in list(self.specials.keys()):
                self.specials[k](self, k, self.store[k])
        self._show_extensions()

    def payload(self, name):
        return self.store['__payload'][name]

    def __setitem__(self, name, value):
        self.store['__payload'][name] = value

    def __getitem__(self, name):
        return self.store['__payload'][name]

    def _writeln(self, msg):
        self.stdout.write(msg + '\n')