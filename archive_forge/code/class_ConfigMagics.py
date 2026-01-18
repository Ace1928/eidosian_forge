import re
from IPython.core.error import UsageError
from IPython.core.magic import Magics, magics_class, line_magic
from logging import error
@magics_class
class ConfigMagics(Magics):

    def __init__(self, shell):
        super(ConfigMagics, self).__init__(shell)
        self.configurables = []

    @line_magic
    def config(self, s):
        """configure IPython

            %config Class[.trait=value]

        This magic exposes most of the IPython config system. Any
        Configurable class should be able to be configured with the simple
        line::

            %config Class.trait=value

        Where `value` will be resolved in the user's namespace, if it is an
        expression or variable name.

        Examples
        --------

        To see what classes are available for config, pass no arguments::

            In [1]: %config
            Available objects for config:
                AliasManager
                DisplayFormatter
                HistoryManager
                IPCompleter
                LoggingMagics
                MagicsManager
                OSMagics
                PrefilterManager
                ScriptMagics
                TerminalInteractiveShell

        To view what is configurable on a given class, just pass the class
        name::

            In [2]: %config LoggingMagics
            LoggingMagics(Magics) options
            ---------------------------
            LoggingMagics.quiet=<Bool>
                Suppress output of log state when logging is enabled
                Current: False

        but the real use is in setting values::

            In [3]: %config LoggingMagics.quiet = True

        and these values are read from the user_ns if they are variables::

            In [4]: feeling_quiet=False

            In [5]: %config LoggingMagics.quiet = feeling_quiet

        """
        from traitlets.config.loader import Config
        configurables = sorted(set([c for c in self.shell.configurables if c.__class__.class_traits(config=True)]), key=lambda x: x.__class__.__name__)
        classnames = [c.__class__.__name__ for c in configurables]
        line = s.strip()
        if not line:
            print('Available objects for config:')
            for name in classnames:
                print('   ', name)
            return
        elif line in classnames:
            c = configurables[classnames.index(line)]
            cls = c.__class__
            help = cls.class_get_help(c)
            help = re.sub(re.compile('^--', re.MULTILINE), '', help)
            print(help)
            return
        elif reg.match(line):
            cls, attr = line.split('.')
            return getattr(configurables[classnames.index(cls)], attr)
        elif '=' not in line:
            msg = 'Invalid config statement: %r, should be `Class.trait = value`.'
            ll = line.lower()
            for classname in classnames:
                if ll == classname.lower():
                    msg = msg + '\nDid you mean %s (note the case)?' % classname
                    break
            raise UsageError(msg % line)
        cfg = Config()
        exec('cfg.' + line, self.shell.user_ns, locals())
        for configurable in configurables:
            try:
                configurable.update_config(cfg)
            except Exception as e:
                error(e)