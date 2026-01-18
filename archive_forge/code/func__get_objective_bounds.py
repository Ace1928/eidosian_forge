import dataclasses
import datetime
import enum
from typing import Dict, Iterable, List, Optional, overload
from ortools.gscip import gscip_pb2
from ortools.math_opt import result_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.solvers import osqp_pb2
def _get_objective_bounds(result_proto: result_pb2.SolveResultProto) -> result_pb2.ObjectiveBoundsProto:
    if result_proto.termination.HasField('objective_bounds'):
        return result_proto.termination.objective_bounds
    return result_pb2.ObjectiveBoundsProto(primal_bound=result_proto.solve_stats.best_primal_bound, dual_bound=result_proto.solve_stats.best_dual_bound)