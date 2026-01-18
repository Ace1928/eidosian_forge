import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.examples.cstr.run_openloop import run_cstr_openloop
def _get_input_sequence(self):
    input_sequence = mpc.TimeSeriesData({'flow_in[*]': [0.1, 1.0, 0.7, 0.9]}, [0.0, 3.0, 6.0, 10.0])
    return mpc.data.convert.series_to_interval(input_sequence)