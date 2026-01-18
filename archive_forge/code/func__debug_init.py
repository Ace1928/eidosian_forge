import timeit
from abc import abstractmethod, ABCMeta
from collections import namedtuple, OrderedDict
import inspect
from pprint import pformat
from numba.core.compiler_lock import global_compiler_lock
from numba.core import errors, config, transforms, utils
from numba.core.tracing import event
from numba.core.postproc import PostProcessor
from numba.core.ir_utils import enforce_no_dels, legalize_single_scope
import numba.core.event as ev
def _debug_init(self):

    def parse(conf_item):
        print_passes = []
        if conf_item != 'none':
            if conf_item == 'all':
                print_passes = [x.name() for x, _ in self.passes]
            else:
                splitted = conf_item.split(',')
                print_passes = [x.strip() for x in splitted]
        return print_passes
    ret = (parse(config.DEBUG_PRINT_AFTER), parse(config.DEBUG_PRINT_BEFORE), parse(config.DEBUG_PRINT_WRAP))
    return ret