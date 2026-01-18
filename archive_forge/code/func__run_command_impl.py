import argparse
import gc
import logging
import os
import sys
import traceback
import types
import time
import json
from pyomo.common.deprecation import deprecated
from pyomo.common.log import is_debug_set
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.fileutils import import_file
from pyomo.common.tee import capture_output
from pyomo.common.dependencies import (
from pyomo.common.collections import Bunch
from pyomo.opt import ProblemFormat
from pyomo.opt.base import SolverFactory
from pyomo.opt.parallel import SolverManagerFactory
from pyomo.dataportal import DataPortal
from pyomo.scripting.interface import (
from pyomo.core import Model, TransformationFactory, Suffix, display
def _run_command_impl(command, parser, args, name, data, options):
    retval = None
    errorcode = 0
    pcount = options.runtime.profile_count
    if pcount > 0:
        try:
            try:
                import cProfile as profile
            except ImportError:
                import profile
            import pstats
        except ImportError:
            raise ValueError("Cannot use the 'profile' option: the Python 'profile' or 'pstats' package cannot be imported!")
        tfile = TempfileManager.create_tempfile(suffix='.profile')
        tmp = profile.runctx(command.__name__ + '(options=options,parser=parser)', command.__globals__, locals(), tfile)
        p = pstats.Stats(tfile).strip_dirs()
        p.sort_stats('time', 'cumulative')
        p = p.print_stats(pcount)
        p.print_callers(pcount)
        p.print_callees(pcount)
        p = p.sort_stats('cumulative', 'calls')
        p.print_stats(pcount)
        p.print_callers(pcount)
        p.print_callees(pcount)
        p = p.sort_stats('calls')
        p.print_stats(pcount)
        p.print_callers(pcount)
        p.print_callees(pcount)
        retval = tmp
    else:
        try:
            retval = command(options=options, parser=parser)
        except SystemExit:
            err = sys.exc_info()[1]
            if __debug__ and (options.runtime.logging == 'debug' or options.runtime.catch_errors):
                sys.exit(0)
            print('Exiting %s: %s' % (name, str(err)))
            errorcode = err.code
        except Exception:
            err = sys.exc_info()[1]
            if __debug__ and (options.runtime.logging == 'debug' or options.runtime.catch_errors):
                raise
            if not options.model is None and (not options.model.save_file is None):
                model = 'model ' + options.model.save_file
            else:
                model = 'model'
            global filter_excepthook
            if filter_excepthook:
                action = 'loading'
            else:
                action = 'running'
            msg = 'Unexpected exception while %s %s:\n    ' % (action, model)
            errStr = str(err)
            if type(err) == KeyError and errStr != 'None':
                errStr = str(err).replace('\\n', '\n')[1:-1]
            logger.error(msg + errStr, extra={'cleandoc': False})
            errorcode = 1
    return (retval, errorcode)