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
class CompilerPass(metaclass=ABCMeta):
    """ The base class for all compiler passes.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        self._analysis = None
        self._pass_id = None

    @classmethod
    def name(cls):
        """
        Returns the name of the pass
        """
        return cls._name

    @property
    def pass_id(self):
        """
        The ID of the pass
        """
        return self._pass_id

    @pass_id.setter
    def pass_id(self, val):
        """
        Sets the ID of the pass
        """
        self._pass_id = val

    @property
    def analysis(self):
        """
        Analysis data for the pass
        """
        return self._analysis

    @analysis.setter
    def analysis(self, val):
        """
        Set the analysis data for the pass
        """
        self._analysis = val

    def run_initialization(self, *args, **kwargs):
        """
        Runs the initialization sequence for the pass, will run before
        `run_pass`.
        """
        return False

    @abstractmethod
    def run_pass(self, *args, **kwargs):
        """
        Runs the pass itself. Must return True/False depending on whether
        statement level modification took place.
        """
        pass

    def run_finalizer(self, *args, **kwargs):
        """
        Runs the initialization sequence for the pass, will run before
        `run_pass`.
        """
        return False

    def get_analysis_usage(self, AU):
        """ Override to set analysis usage
        """
        pass

    def get_analysis(self, pass_name):
        """
        Gets the analysis from a given pass
        """
        return self._analysis[pass_name]