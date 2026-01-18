import json
from typing import List, Optional, Tuple
from google.protobuf import json_format
import requests
from ortools.service.v1 import optimization_pb2
from ortools.math_opt import rpc_pb2
from ortools.math_opt.python import mathopt
from ortools.math_opt.python.ipc import proto_converter
def _build_json_payload(model: mathopt.Model, solver_type: mathopt.SolverType, params: Optional[mathopt.SolveParameters], model_params: Optional[mathopt.ModelSolveParameters]):
    """Builds a JSON payload.

    Args:
      model: The optimization model.
      solver_type: The underlying solver to use.
      params: Optional configuration of the underlying solver.
      model_params: Optional configuration of the solver that is model specific.

    Returns:
      A JSON object with a MathOpt model and corresponding parameters.

    Raises:
      SerializationError: If building the OR API proto is not successful or
        deserializing to JSON fails.
    """
    params = params or mathopt.SolveParameters()
    model_params = model_params or mathopt.ModelSolveParameters()
    try:
        request = rpc_pb2.SolveRequest(model=model.export_model(), solver_type=solver_type.value, parameters=params.to_proto(), model_parameters=model_params.to_proto())
        api_request = proto_converter.convert_request(request)
    except ValueError as err:
        raise ValueError from err
    return json.loads(json_format.MessageToJson(api_request))