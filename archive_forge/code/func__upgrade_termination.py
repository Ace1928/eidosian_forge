import dataclasses
import datetime
import enum
from typing import Dict, Iterable, List, Optional, overload
from ortools.gscip import gscip_pb2
from ortools.math_opt import result_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.solvers import osqp_pb2
def _upgrade_termination(result_proto: result_pb2.SolveResultProto) -> result_pb2.TerminationProto:
    return result_pb2.TerminationProto(reason=result_proto.termination.reason, limit=result_proto.termination.limit, detail=result_proto.termination.detail, problem_status=_get_problem_status(result_proto), objective_bounds=_get_objective_bounds(result_proto))