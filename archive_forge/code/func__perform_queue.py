import logging
import os
import re
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.opt import SolverFactory, SolverManagerFactory, OptSolver
from pyomo.opt.parallel.manager import ActionManagerError, ActionStatus
from pyomo.opt.parallel.async_solver import AsynchronousSolverManager
from pyomo.core.base import Block
import pyomo.neos.kestrel
def _perform_queue(self, ah, *args, **kwds):
    """
        Perform the queue operation.  This method returns the ActionHandle,
        and the ActionHandle status indicates whether the queue was successful.
        """
    solver = kwds.pop('solver', kwds.pop('opt', None))
    if solver is None:
        raise ActionManagerError("No solver passed to %s, use keyword option 'solver'" % type(self).__name__)
    if not isinstance(solver, str):
        solver_name = solver.name
        if solver_name == 'asl':
            solver_name = os.path.basename(solver.executable())
    else:
        solver_name = solver
        solver = None
    user_solver_options = {}
    if solver is not None:
        user_solver_options.update(solver.options)
    _options = kwds.pop('options', {})
    if isinstance(_options, str):
        _options = OptSolver._options_string_to_dict(_options)
    user_solver_options.update(_options)
    user_solver_options.update(OptSolver._options_string_to_dict(kwds.pop('options_string', '')))
    if user_solver_options.get('timelimit', 0) is None:
        del user_solver_options['timelimit']
    opt = SolverFactory('_neos')
    opt._presolve(*args, **kwds)
    if len(self._solvers) == 0:
        for name in self.kestrel.solvers():
            if name.endswith('AMPL'):
                self._solvers[name[:-5].lower()] = name[:-5]
    if solver_name not in self._solvers:
        raise ActionManagerError("Solver '%s' is not recognized by NEOS. Solver names recognized:\n%s" % (solver_name, str(sorted(self._solvers.keys()))))
    neos_sname = self._solvers[solver_name].lower()
    os.environ['kestrel_options'] = 'solver=%s' % self._solvers[solver_name]
    solver_options = {}
    for key in opt.options:
        solver_options[key] = opt.options[key]
    solver_options.update(user_solver_options)
    options = opt._get_options_string(solver_options)
    if not options == '':
        os.environ[neos_sname + '_options'] = options
    xml = self.kestrel.formXML(opt._problem_files[0])
    jobNumber, password = self.kestrel.submit(xml)
    ah.job = jobNumber
    ah.password = password
    del os.environ['kestrel_options']
    try:
        del os.environ[neos_sname + '_options']
    except:
        pass
    self._ah[jobNumber] = ah
    self._neos_log[jobNumber] = (0, '')
    self._opt_data[jobNumber] = (opt, opt._smap_id, opt._load_solutions, opt._select_index, opt._default_variable_value)
    self._args[jobNumber] = args
    return ah