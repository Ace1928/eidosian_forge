import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.examples.cstr.run_mpc import get_steady_state_data, run_cstr_mpc
def _get_setpoint_data(self):
    setpoint_data = mpc.ScalarData({'flow_in[*]': 1.2})
    return get_steady_state_data(setpoint_data)