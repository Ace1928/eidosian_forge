import logging
from io import StringIO
from pyomo.common.gc_manager import PauseGC
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter, WriterFactory
from pyomo.core.base import (
from pyomo.repn import generate_standard_repn
def _printSOS(self, symbol_map, labeler, variable_symbol_map, soscondata, output_file):
    if hasattr(soscondata, 'get_items'):
        sos_items = list(soscondata.get_items())
    else:
        sos_items = list(soscondata.items())
    if len(sos_items) == 0:
        return
    output_file.write('SOS\n')
    level = soscondata.level
    output_file.write(' S%d %s\n' % (level, symbol_map.getSymbol(soscondata, labeler)))
    sos_template_string = '    %s %' + self._precision_string + '\n'
    for vardata, weight in sos_items:
        weight = _get_bound(weight)
        if weight < 0:
            raise ValueError('Cannot use negative weight %f for variable %s is special ordered set %s ' % (weight, vardata.name, soscondata.name))
        if vardata.fixed:
            raise RuntimeError("SOSConstraint '%s' includes a fixed variable '%s'. This is currently not supported. Deactivate this constraint in order to proceed." % (soscondata.name, vardata.name))
        self._referenced_variable_ids[id(vardata)] = vardata
        output_file.write(sos_template_string % (variable_symbol_map.getSymbol(vardata), weight))