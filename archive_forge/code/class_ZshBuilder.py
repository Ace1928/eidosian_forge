import getopt
import inspect
import itertools
from types import MethodType
from typing import Dict, List, Set
from twisted.python import reflect, usage, util
from twisted.python.compat import ioType
class ZshBuilder:
    """
    Constructs zsh code that will complete options for a given usage.Options
    instance, possibly including a list of subcommand names.

    Completions for options to subcommands won't be generated because this
    class will never be used if the user is completing options for a specific
    subcommand. (See L{ZshSubcommandBuilder} below)

    @type options: L{twisted.python.usage.Options}
    @ivar options: The L{twisted.python.usage.Options} instance defined for this
        command.

    @type cmdName: C{str}
    @ivar cmdName: The name of the command we're generating completions for.

    @type file: C{file}
    @ivar file: The C{file} to write the completion function to.  The C{file}
        must have L{bytes} I/O semantics.
    """

    def __init__(self, options, cmdName, file):
        self.options = options
        self.cmdName = cmdName
        self.file = file

    def write(self, genSubs=True):
        """
        Generate the completion function and write it to the output file
        @return: L{None}

        @type genSubs: C{bool}
        @param genSubs: Flag indicating whether or not completions for the list
            of subcommand should be generated. Only has an effect
            if the C{subCommands} attribute has been defined on the
            L{twisted.python.usage.Options} instance.
        """
        if genSubs and getattr(self.options, 'subCommands', None) is not None:
            gen = ZshArgumentsGenerator(self.options, self.cmdName, self.file)
            gen.extraActions.insert(0, SubcommandAction())
            gen.write()
            self.file.write(b'local _zsh_subcmds_array\n_zsh_subcmds_array=(\n')
            for cmd, short, parser, desc in self.options.subCommands:
                self.file.write(b'"' + cmd.encode('utf-8') + b':' + desc.encode('utf-8') + b'"\n')
            self.file.write(b')\n\n')
            self.file.write(b'_describe "sub-command" _zsh_subcmds_array\n')
        else:
            gen = ZshArgumentsGenerator(self.options, self.cmdName, self.file)
            gen.write()