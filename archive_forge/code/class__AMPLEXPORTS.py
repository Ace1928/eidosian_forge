import logging
import types
import weakref
from pyomo.common.pyomo_typing import overload
from ctypes import (
from pyomo.common.autoslots import AutoSlots
from pyomo.common.fileutils import find_library
from pyomo.core.expr.numvalue import (
import pyomo.core.expr as EXPR
from pyomo.core.base.component import Component
from pyomo.core.base.units_container import units
class _AMPLEXPORTS(Structure):
    """Mock up the AmplExports structure from AMPL's funcadd.h

    The only thing we really need here is to be able to override the
    Addfunc function pointer to point back into Python so we can
    intercept the registration of external AMPL functions.  Ideally, we
    would populate the other function pointers as well, but that is
    trickier than it sounds, and at least so far is completely unneeded.
    """
    AMPLFUNC = CFUNCTYPE(c_double, POINTER(_ARGLIST))
    ADDFUNC = CFUNCTYPE(None, c_char_p, AMPLFUNC, c_int, c_int, c_void_p, POINTER(_AMPLEXPORTS))
    RANDSEEDSETTER = CFUNCTYPE(None, c_void_p, c_ulong)
    ADDRANDINIT = CFUNCTYPE(None, POINTER(_AMPLEXPORTS), RANDSEEDSETTER, c_void_p)
    ATRESET = CFUNCTYPE(None, POINTER(_AMPLEXPORTS), c_void_p, c_void_p)
    _fields_ = [('StdErr', c_void_p), ('Addfunc', ADDFUNC), ('ASLdate', c_long), ('FprintF', c_void_p), ('PrintF', c_void_p), ('SprintF', c_void_p), ('VfprintF', c_void_p), ('VsprintF', c_void_p), ('Strtod', c_void_p), ('Crypto', c_void_p), ('asl', c_char_p), ('AtExit', c_void_p), ('AtReset', ATRESET), ('Tempmem', c_void_p), ('Add_table_handler', c_void_p), ('Private', c_char_p), ('Qsortv', c_void_p), ('StdIn', c_void_p), ('StdOut', c_void_p), ('Clearerr', c_void_p), ('Fclose', c_void_p), ('Fdopen', c_void_p), ('Feof', c_void_p), ('Ferror', c_void_p), ('Fflush', c_void_p), ('Fgetc', c_void_p), ('Fgets', c_void_p), ('Fileno', c_void_p), ('Fopen', c_void_p), ('Fputc', c_void_p), ('Fputs', c_void_p), ('Fread', c_void_p), ('Freopen', c_void_p), ('Fscanf', c_void_p), ('Fseek', c_void_p), ('Ftell', c_void_p), ('Fwrite', c_void_p), ('Pclose', c_void_p), ('Perror', c_void_p), ('Popen', c_void_p), ('Puts', c_void_p), ('Rewind', c_void_p), ('Scanf', c_void_p), ('Setbuf', c_void_p), ('Setvbuf', c_void_p), ('Sscanf', c_void_p), ('Tempnam', c_void_p), ('Tmpfile', c_void_p), ('Tmpnam', c_void_p), ('Ungetc', c_void_p), ('AI', c_void_p), ('Getenv', c_void_p), ('Breakfunc', c_void_p), ('Breakarg', c_char_p), ('SnprintF', c_void_p), ('VsnprintF', c_void_p), ('Addrand', c_void_p), ('Addrandinit', ADDRANDINIT)]