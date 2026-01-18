import types
from typing import Callable, Optional
from ortools.math_opt import parameters_pb2
from ortools.math_opt.core.python import solver
from ortools.math_opt.python import callback
from ortools.math_opt.python import compute_infeasible_subsystem_result
from ortools.math_opt.python import message_callback
from ortools.math_opt.python import model
from ortools.math_opt.python import model_parameters
from ortools.math_opt.python import parameters
from ortools.math_opt.python import result
from pybind11_abseil.status import StatusNotOk
class IncrementalSolver:
    """Solve an optimization multiple times, with modifications between solves.

    Prefer calling simply solve() above in most cases when incrementalism is not
    needed.

    Thread-safety: The __init__(), solve() methods must not be called while
    modifying the Model (adding variables...). The user is expected to use proper
    synchronization primitives to serialize changes to the model and the use of
    this object. Note though that it is safe to call methods from different
    IncrementalSolver instances on the same Model concurrently. The solve() method
    must not be called concurrently on different threads for the same
    IncrementalSolver. Some solvers may add more restriction regarding
    threading. Please see to SolverType::XXX documentation for details.

    This class references some resources that are freed when it is garbage
    collected (which should usually happen when the last reference is lost). In
    particular, it references some C++ objects. Although it is not mandatory, it
    is recommended to free those as soon as possible. To do so it is possible to
    use this class in the `with` statement:

    with IncrementalSolver(model, SolverType.GLOP) as solver:
      ...

    When it is not possible to use `with`, the close() method can be called.
    """

    def __init__(self, opt_model: model.Model, solver_type: parameters.SolverType):
        self._model = opt_model
        self._solver_type = solver_type
        self._update_tracker = self._model.add_update_tracker()
        try:
            self._proto_solver = solver.new(solver_type.value, self._model.export_model(), parameters_pb2.SolverInitializerProto())
        except StatusNotOk as e:
            raise RuntimeError(str(e)) from None
        self._closed = False

    def solve(self, *, params: Optional[parameters.SolveParameters]=None, model_params: Optional[model_parameters.ModelSolveParameters]=None, msg_cb: Optional[message_callback.SolveMessageCallback]=None, callback_reg: Optional[callback.CallbackRegistration]=None, cb: Optional[SolveCallback]=None) -> result.SolveResult:
        """Solves the current optimization model.

        Args:
          params: The non-model specific solve parameters.
          model_params: The model specific solve parameters.
          msg_cb: An optional callback for solver messages.
          callback_reg: The parameters controlling when cb is called.
          cb: An optional callback for LP/MIP events.

        Returns:
          The result of the solve.

        Raises:
          RuntimeError: If called after being closed, or on a solve error.
        """
        if self._closed:
            raise RuntimeError('the solver is closed')
        update = self._update_tracker.export_update()
        if update is not None:
            try:
                if not self._proto_solver.update(update):
                    self._proto_solver = solver.new(self._solver_type.value, self._model.export_model(), parameters_pb2.SolverInitializerProto())
            except StatusNotOk as e:
                raise RuntimeError(str(e)) from None
            self._update_tracker.advance_checkpoint()
        params = params or parameters.SolveParameters()
        model_params = model_params or model_parameters.ModelSolveParameters()
        callback_reg = callback_reg or callback.CallbackRegistration()
        proto_cb = None
        if cb is not None:
            proto_cb = lambda x: cb(callback.parse_callback_data(x, self._model)).to_proto()
        try:
            result_proto = self._proto_solver.solve(params.to_proto(), model_params.to_proto(), msg_cb, callback_reg.to_proto(), proto_cb, None)
        except StatusNotOk as e:
            raise RuntimeError(str(e)) from None
        return result.parse_solve_result(result_proto, self._model)

    def close(self) -> None:
        """Closes this solver, freeing all its resources.

        This is optional, the code is correct without calling this function. See the
        class documentation for details.

        After a solver has been closed, it can't be used anymore. Prefer using the
        context manager API when possible instead of calling close() directly:

        with IncrementalSolver(model, SolverType.GLOP) as solver:
          ...
        """
        if self._closed:
            return
        self._closed = True
        del self._model
        del self._solver_type
        del self._update_tracker
        del self._proto_solver

    def __enter__(self) -> 'IncrementalSolver':
        """Returns the solver itself."""
        return self

    def __exit__(self, exc_type: Optional[type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[types.TracebackType]) -> None:
        """Closes the solver."""
        self.close()