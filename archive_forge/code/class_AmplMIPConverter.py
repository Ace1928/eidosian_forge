import os.path
import subprocess
from pyomo.common.errors import ApplicationError
import pyomo.common
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt.base import ProblemFormat, ConverterError
from pyomo.opt.base.convert import ProblemConverterFactory
@ProblemConverterFactory.register('ampl')
class AmplMIPConverter(object):

    def can_convert(self, from_type, to_type):
        """Returns true if this object supports the specified conversion"""
        if not pyomo.common.Executable('ampl'):
            return False
        if from_type == ProblemFormat.mod and to_type == ProblemFormat.nl:
            return True
        if from_type == ProblemFormat.mod and to_type == ProblemFormat.mps:
            return True
        return False

    def apply(self, *args, **kwargs):
        """Convert an instance of one type into another"""
        if not isinstance(args[2], str):
            raise ConverterError('Can only apply ampl to convert file data')
        _exec = pyomo.common.Executable('ampl')
        if not _exec:
            raise ConverterError("The 'ampl' executable cannot be found")
        script_filename = TempfileManager.create_tempfile(suffix='.ampl')
        if args[1] == ProblemFormat.nl:
            output_filename = TempfileManager.create_tempfile(suffix='.nl')
        else:
            output_filename = TempfileManager.create_tempfile(suffix='.mps')
        cmd = [_exec.path(), script_filename]
        OUTPUT = open(script_filename, 'w')
        OUTPUT.write('#\n')
        OUTPUT.write('# AMPL script for converting the following files\n')
        OUTPUT.write('#\n')
        if len(args[2:]) == 1:
            OUTPUT.write('model ' + args[2] + ';\n')
        else:
            OUTPUT.write('model ' + args[2] + ';\n')
            OUTPUT.write('data ' + args[3] + ';\n')
        abs_ofile = os.path.abspath(output_filename)
        if args[1] == ProblemFormat.nl:
            OUTPUT.write('write g' + abs_ofile[:-3] + ';\n')
        else:
            OUTPUT.write('write m' + abs_ofile[:-4] + ';\n')
        OUTPUT.close()
        output = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        if not os.path.exists(output_filename):
            raise ApplicationError("Problem launching 'ampl' to create '%s': %s" % (output_filename, output.stdout))
        return ((output_filename,), None)