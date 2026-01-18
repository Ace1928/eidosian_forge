import inspect
import re
import warnings
from typing import Any, Dict
import torch
from torch.testing import make_tensor
Find a dict of parameters that minimizes the target function using
    the initial dict of parameters and a step function that progresses
    a specified parameter in a dict of parameters.

    Parameters
    ----------
    target_func (callable): a functional with the signature
      ``target_func(parameters: dict) -> float``
    initial_parameters (dict): a set of parameters used as an initial
      value to the minimization process.
    reference_parameters (dict): a set of parameters used as an
      reference value with respect to which the speed up is computed.
    step_func (callable): a functional with the signature
      ``step_func(parameter_name:str, parameter_value:int, direction:int, parameters:dict) -> int``
      that increments or decrements (when ``direction`` is positive or
      negative, respectively) the parameter with given name and value.
      When return value is equal to ``parameter_value``, it means that
      no step along the given direction can be made.

    Returns
    -------
    parameters (dict): a set of parameters that minimizes the target
      function.
    speedup_incr (float): a speedup change given in percentage.
    timing (float): the value of the target function at the parameters.
    sensitivity_message (str): a message containing sensitivity.
      information of parameters around the target function minimizer.
    