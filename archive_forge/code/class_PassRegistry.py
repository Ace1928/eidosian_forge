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
class PassRegistry(object):
    """
    Pass registry singleton class.
    """
    _id = 0
    _registry = dict()

    def register(self, mutates_CFG, analysis_only):

        def make_festive(pass_class):
            assert not self.is_registered(pass_class)
            assert not self._does_pass_name_alias(pass_class.name())
            pass_class.pass_id = self._id
            self._id += 1
            self._registry[pass_class] = pass_info(pass_class(), mutates_CFG, analysis_only)
            return pass_class
        return make_festive

    def is_registered(self, clazz):
        return clazz in self._registry.keys()

    def get(self, clazz):
        assert self.is_registered(clazz)
        return self._registry[clazz]

    def _does_pass_name_alias(self, check):
        for k, v in self._registry.items():
            if v.pass_inst.name == check:
                return True
        return False

    def find_by_name(self, class_name):
        assert isinstance(class_name, str)
        for k, v in self._registry.items():
            if v.pass_inst.name == class_name:
                return v
        else:
            raise ValueError('No pass with name %s is registered' % class_name)

    def dump(self):
        for k, v in self._registry.items():
            print('%s: %s' % (k, v))