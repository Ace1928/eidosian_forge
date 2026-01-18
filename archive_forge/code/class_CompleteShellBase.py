import logging
import stevedore
from cliff import command
class CompleteShellBase(object):
    """base class for bash completion generation
    """

    def __init__(self, name, output):
        self.name = str(name)
        self.output = output

    def write(self, cmdo, data):
        self.output.write(self.get_header())
        self.output.write("  cmds='{0}'\n".format(cmdo))
        for datum in data:
            datum = (datum[0].replace('-', '_'), datum[1])
            self.output.write("  cmds_{0}='{1}'\n".format(*datum))
        self.output.write(self.get_trailer())

    @property
    def escaped_name(self):
        return self.name.replace('-', '_')