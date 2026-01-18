import re
import sys
import time
import logging
import shlex
from pyomo.common import Factory
from pyomo.common.errors import ApplicationError
from pyomo.common.collections import Bunch
from pyomo.opt.base.convert import convert_problem
from pyomo.opt.base.formats import ResultsFormat
import pyomo.opt.base.results
class OptSolver(object):
    """A generic optimization solver"""

    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass

    @property
    def tee(self):
        _raise_ephemeral_error('tee')

    @tee.setter
    def tee(self, val):
        _raise_ephemeral_error('tee')

    @property
    def suffixes(self):
        _raise_ephemeral_error('suffixes')

    @suffixes.setter
    def suffixes(self, val):
        _raise_ephemeral_error('suffixes')

    @property
    def keepfiles(self):
        _raise_ephemeral_error('keepfiles')

    @keepfiles.setter
    def keepfiles(self, val):
        _raise_ephemeral_error('keepfiles')

    @property
    def soln_file(self):
        _raise_ephemeral_error('soln_file')

    @soln_file.setter
    def soln_file(self, val):
        _raise_ephemeral_error('soln_file')

    @property
    def log_file(self):
        _raise_ephemeral_error('log_file')

    @log_file.setter
    def log_file(self, val):
        _raise_ephemeral_error('log_file')

    @property
    def symbolic_solver_labels(self):
        _raise_ephemeral_error('symbolic_solver_labels')

    @symbolic_solver_labels.setter
    def symbolic_solver_labels(self, val):
        _raise_ephemeral_error('symbolic_solver_labels')

    @property
    def warm_start_solve(self):
        _raise_ephemeral_error('warm_start_solve', keyword=' (warmstart)')

    @warm_start_solve.setter
    def warm_start_solve(self, val):
        _raise_ephemeral_error('warm_start_solve', keyword=' (warmstart)')

    @property
    def warm_start_file_name(self):
        _raise_ephemeral_error('warm_start_file_name', keyword=' (warmstart_file)')

    @warm_start_file_name.setter
    def warm_start_file_name(self, val):
        _raise_ephemeral_error('warm_start_file_name', keyword=' (warmstart_file)')

    def __init__(self, **kwds):
        """Constructor"""
        if 'type' in kwds:
            self.type = kwds['type']
        else:
            raise ValueError("Expected option 'type' for OptSolver constructor")
        if 'name' in kwds:
            self.name = kwds['name']
        else:
            self.name = self.type
        if 'doc' in kwds:
            self._doc = kwds['doc']
        elif self.type is None:
            self._doc = ''
        elif self.name == self.type:
            self._doc = '%s OptSolver' % self.name
        else:
            self._doc = '%s OptSolver (type %s)' % (self.name, self.type)
        self.options = Bunch()
        if 'options' in kwds and (not kwds['options'] is None):
            for key in kwds['options']:
                setattr(self.options, key, kwds['options'][key])
        self._smap_id = None
        self._load_solutions = True
        self._select_index = 0
        self._report_timing = False
        self._suffixes = []
        self._log_file = None
        self._soln_file = None
        self._default_variable_value = None
        self._assert_available = False
        self._problem_format = None
        self._valid_problem_formats = []
        self._results_format = None
        self._valid_result_formats = {}
        self._results_reader = None
        self._problem = None
        self._problem_files = None
        self._metasolver = False
        self._version = None
        self._allow_callbacks = False
        self._callback = {}
        self._capabilities = Bunch()

    @staticmethod
    def _options_string_to_dict(istr):
        ans = {}
        istr = istr.strip()
        if not istr:
            return ans
        if istr[0] == "'" or istr[0] == '"':
            istr = eval(istr)
        tokens = shlex.split(istr)
        for token in tokens:
            index = token.find('=')
            if index == -1:
                raise ValueError("Solver options must have the form option=value: '%s'" % istr)
            try:
                val = eval(token[index + 1:])
            except:
                val = token[index + 1:]
            ans[token[:index]] = val
        return ans

    def default_variable_value(self):
        return self._default_variable_value

    def __bool__(self):
        return self.available()

    def version(self):
        """
        Returns a 4-tuple describing the solver executable version.
        """
        if self._version is None:
            self._version = self._get_version()
        return self._version

    def _get_version(self):
        return None

    def problem_format(self):
        """
        Returns the current problem format.
        """
        return self._problem_format

    def set_problem_format(self, format):
        """
        Set the current problem format (if it's valid) and update
        the results format to something valid for this problem format.
        """
        if format in self._valid_problem_formats:
            self._problem_format = format
        else:
            raise ValueError('%s is not a valid problem format for solver plugin %s' % (format, self))
        self._results_format = self._default_results_format(self._problem_format)

    def results_format(self):
        """
        Returns the current results format.
        """
        return self._results_format

    def set_results_format(self, format):
        """
        Set the current results format (if it's valid for the current
        problem format).
        """
        if self._problem_format in self._valid_results_formats and format in self._valid_results_formats[self._problem_format]:
            self._results_format = format
        else:
            raise ValueError('%s is not a valid results format for problem format %s with solver plugin %s' % (format, self._problem_format, self))

    def has_capability(self, cap):
        """
        Returns a boolean value representing whether a solver supports
        a specific feature. Defaults to 'False' if the solver is unaware
        of an option. Expects a string.

        Example:
        # prints True if solver supports sos1 constraints, and False otherwise
        print(solver.has_capability('sos1')

        # prints True is solver supports 'feature', and False otherwise
        print(solver.has_capability('feature')

        Parameters
        ----------
        cap: str
            The feature

        Returns
        -------
        val: bool
            Whether or not the solver has the specified capability.
        """
        if not isinstance(cap, str):
            raise TypeError("Expected argument to be of type '%s', not '%s'." % (type(str()), type(cap)))
        else:
            val = self._capabilities[str(cap)]
            if val is None:
                return False
            else:
                return val

    def available(self, exception_flag=True):
        """True if the solver is available"""
        return True

    def license_is_valid(self):
        """True if the solver is present and has a valid license (if applicable)"""
        return True

    def warm_start_capable(self):
        """True is the solver can accept a warm-start solution"""
        return False

    def solve(self, *args, **kwds):
        """Solve the problem"""
        self.available(exception_flag=True)
        from pyomo.core.base.block import _BlockData
        import pyomo.core.base.suffix
        from pyomo.core.kernel.block import IBlock
        import pyomo.core.kernel.suffix
        _model = None
        for arg in args:
            if isinstance(arg, (_BlockData, IBlock)):
                if isinstance(arg, _BlockData):
                    if not arg.is_constructed():
                        raise RuntimeError('Attempting to solve model=%s with unconstructed component(s)' % (arg.name,))
                _model = arg
                if isinstance(arg, _BlockData):
                    model_suffixes = list((name for name, comp in pyomo.core.base.suffix.active_import_suffix_generator(arg)))
                else:
                    assert isinstance(arg, IBlock)
                    model_suffixes = list((comp.storage_key for comp in pyomo.core.kernel.suffix.import_suffix_generator(arg, active=True, descend_into=False)))
                if len(model_suffixes) > 0:
                    kwds_suffixes = kwds.setdefault('suffixes', [])
                    for name in model_suffixes:
                        if name not in kwds_suffixes:
                            kwds_suffixes.append(name)
        orig_options = self.options
        self.options = Bunch()
        self.options.update(orig_options)
        self.options.update(kwds.pop('options', {}))
        self.options.update(self._options_string_to_dict(kwds.pop('options_string', '')))
        try:
            initial_time = time.time()
            self._presolve(*args, **kwds)
            presolve_completion_time = time.time()
            if self._report_timing:
                print('      %6.2f seconds required for presolve' % (presolve_completion_time - initial_time))
            if not _model is None:
                self._initialize_callbacks(_model)
            _status = self._apply_solver()
            if hasattr(self, '_transformation_data'):
                del self._transformation_data
            if not hasattr(_status, 'rc'):
                logger.warning('Solver (%s) did not return a solver status code.\nThis is indicative of an internal solver plugin error.\nPlease report this to the Pyomo developers.')
            elif _status.rc:
                logger.error('Solver (%s) returned non-zero return code (%s)' % (self.name, _status.rc))
                if self._tee:
                    logger.error('See the solver log above for diagnostic information.')
                elif hasattr(_status, 'log') and _status.log:
                    logger.error('Solver log:\n' + str(_status.log))
                raise ApplicationError('Solver (%s) did not exit normally' % self.name)
            solve_completion_time = time.time()
            if self._report_timing:
                print('      %6.2f seconds required for solver' % (solve_completion_time - presolve_completion_time))
            result = self._postsolve()
            result._smap_id = self._smap_id
            result._smap = None
            if _model:
                if isinstance(_model, IBlock):
                    if len(result.solution) == 1:
                        result.solution(0).symbol_map = getattr(_model, '._symbol_maps')[result._smap_id]
                        result.solution(0).default_variable_value = self._default_variable_value
                        if self._load_solutions:
                            _model.load_solution(result.solution(0))
                    else:
                        assert len(result.solution) == 0
                    assert len(getattr(_model, '._symbol_maps')) == 1
                    delattr(_model, '._symbol_maps')
                    del result._smap_id
                    if self._load_solutions and len(result.solution) == 0:
                        logger.error('No solution is available')
                elif self._load_solutions:
                    _model.solutions.load_from(result, select=self._select_index, default_variable_value=self._default_variable_value)
                    result._smap_id = None
                    result.solution.clear()
                else:
                    result._smap = _model.solutions.symbol_map[self._smap_id]
                    _model.solutions.delete_symbol_map(self._smap_id)
            postsolve_completion_time = time.time()
            if self._report_timing:
                print('      %6.2f seconds required for postsolve' % (postsolve_completion_time - solve_completion_time))
        finally:
            self.options = orig_options
        return result

    def _presolve(self, *args, **kwds):
        self._log_file = kwds.pop('logfile', None)
        self._soln_file = kwds.pop('solnfile', None)
        self._select_index = kwds.pop('select', 0)
        self._load_solutions = kwds.pop('load_solutions', True)
        self._timelimit = kwds.pop('timelimit', None)
        self._report_timing = kwds.pop('report_timing', False)
        self._tee = kwds.pop('tee', False)
        self._assert_available = kwds.pop('available', True)
        self._suffixes = kwds.pop('suffixes', [])
        self.available()
        if self._problem_format:
            write_start_time = time.time()
            self._problem_files, self._problem_format, self._smap_id = self._convert_problem(args, self._problem_format, self._valid_problem_formats, **kwds)
            total_time = time.time() - write_start_time
            if self._report_timing:
                print('      %6.2f seconds required to write file' % total_time)
        elif len(kwds):
            raise ValueError('Solver=' + self.type + ' passed unrecognized keywords: \n\t' + '\n\t'.join(('%s = %s' % (k, v) for k, v in kwds.items())))
        if type(self._problem_files) in (list, tuple) and (not isinstance(self._problem_files[0], str)):
            self._problem_files = self._problem_files[0]._problem_files()
        if self._results_format is None:
            self._results_format = self._default_results_format(self._problem_format)
        if self._results_format == ResultsFormat.soln:
            self._results_reader = None
        else:
            self._results_reader = pyomo.opt.base.results.ReaderFactory(self._results_format)

    def _initialize_callbacks(self, model):
        """Initialize call-back functions"""
        pass

    def _apply_solver(self):
        """The routine that performs the solve"""
        raise NotImplementedError

    def _postsolve(self):
        """The routine that does solve post-processing"""
        return self.results

    def _convert_problem(self, args, problem_format, valid_problem_formats, **kwds):
        return convert_problem(args, problem_format, valid_problem_formats, self.has_capability, **kwds)

    def _default_results_format(self, prob_format):
        """Returns the default results format for different problem
        formats.
        """
        return ResultsFormat.results

    def reset(self):
        """
        Reset the state of the solver
        """
        pass

    def _get_options_string(self, options=None):
        if options is None:
            options = self.options
        ans = []
        for key in options:
            val = options[key]
            if isinstance(val, str) and ' ' in val:
                ans.append('%s="%s"' % (str(key), str(val)))
            else:
                ans.append('%s=%s' % (str(key), str(val)))
        return ' '.join(ans)

    def set_options(self, istr):
        if isinstance(istr, str):
            istr = self._options_string_to_dict(istr)
        for key in istr:
            if not istr[key] is None:
                setattr(self.options, key, istr[key])

    def set_callback(self, name, callback_fn=None):
        """
        Set the callback function for a named callback.

        A call-back function has the form:

            def fn(solver, model):
                pass

        where 'solver' is the native solver interface object and 'model' is
        a Pyomo model instance object.
        """
        if not self._allow_callbacks:
            raise ApplicationError('Callbacks disabled for solver %s' % self.name)
        if callback_fn is None:
            if name in self._callback:
                del self._callback[name]
        else:
            self._callback[name] = callback_fn

    def config_block(self, init=False):
        from pyomo.scripting.solve_config import default_config_block
        return default_config_block(self, init)[0]