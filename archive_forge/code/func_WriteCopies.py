import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def WriteCopies(self, copies, extra_outputs, part_of_all):
    """Write Makefile code for any 'copies' from the gyp input.

        extra_outputs: a list that will be filled in with any outputs of this action
                       (used to make other pieces dependent on this action)
        part_of_all: flag indicating this target is part of 'all'
        """
    self.WriteLn('### Generated for copy rule.')
    variable = StringToMakefileVariable(self.qualified_target + '_copies')
    outputs = []
    for copy in copies:
        for path in copy['files']:
            path = Sourceify(self.Absolutify(path))
            filename = os.path.split(path)[1]
            output = Sourceify(self.Absolutify(os.path.join(copy['destination'], filename)))
            env = self.GetSortedXcodeEnv()
            output = gyp.xcode_emulation.ExpandEnvVars(output, env)
            path = gyp.xcode_emulation.ExpandEnvVars(path, env)
            self.WriteDoCmd([output], [path], 'copy', part_of_all)
            outputs.append(output)
    self.WriteLn('{} = {}'.format(variable, ' '.join((QuoteSpaces(o) for o in outputs))))
    extra_outputs.append('$(%s)' % variable)
    self.WriteLn()