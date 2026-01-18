import sys
import warnings
from IPython.core import ultratb, compilerop
from IPython.core import magic_arguments
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.core.interactiveshell import DummyMod, InteractiveShell
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.terminal.ipapp import load_default_config
from traitlets import Bool, CBool, Unicode
from IPython.utils.io import ask_yes_no
from typing import Set
class InteractiveShellEmbed(TerminalInteractiveShell):
    dummy_mode = Bool(False)
    exit_msg = Unicode('')
    embedded = CBool(True)
    should_raise = CBool(False)
    display_banner = CBool(True)
    exit_msg = Unicode()
    term_title = Bool(False, help='Automatically set the terminal title').tag(config=True)
    _inactive_locations: Set[str] = set()

    def _disable_init_location(self):
        """Disable the current Instance creation location"""
        InteractiveShellEmbed._inactive_locations.add(self._init_location_id)

    @property
    def embedded_active(self):
        return self._call_location_id not in InteractiveShellEmbed._inactive_locations and self._init_location_id not in InteractiveShellEmbed._inactive_locations

    @embedded_active.setter
    def embedded_active(self, value):
        if value:
            InteractiveShellEmbed._inactive_locations.discard(self._call_location_id)
            InteractiveShellEmbed._inactive_locations.discard(self._init_location_id)
        else:
            InteractiveShellEmbed._inactive_locations.add(self._call_location_id)

    def __init__(self, **kw):
        assert 'user_global_ns' not in kw, 'Key word argument `user_global_ns` has been replaced by `user_module` since IPython 4.0.'
        cls = type(self)
        if cls._instance is None:
            for subclass in cls._walk_mro():
                subclass._instance = self
            cls._instance = self
        clid = kw.pop('_init_location_id', None)
        if not clid:
            frame = sys._getframe(1)
            clid = '%s:%s' % (frame.f_code.co_filename, frame.f_lineno)
        self._init_location_id = clid
        super(InteractiveShellEmbed, self).__init__(**kw)
        sys.excepthook = ultratb.FormattedTB(color_scheme=self.colors, mode=self.xmode, call_pdb=self.pdb)

    def init_sys_modules(self):
        """
        Explicitly overwrite :mod:`IPython.core.interactiveshell` to do nothing.
        """
        pass

    def init_magics(self):
        super(InteractiveShellEmbed, self).init_magics()
        self.register_magics(EmbeddedMagics)

    def __call__(self, header='', local_ns=None, module=None, dummy=None, stack_depth=1, compile_flags=None, **kw):
        """Activate the interactive interpreter.

        __call__(self,header='',local_ns=None,module=None,dummy=None) -> Start
        the interpreter shell with the given local and global namespaces, and
        optionally print a header string at startup.

        The shell can be globally activated/deactivated using the
        dummy_mode attribute. This allows you to turn off a shell used
        for debugging globally.

        However, *each* time you call the shell you can override the current
        state of dummy_mode with the optional keyword parameter 'dummy'. For
        example, if you set dummy mode on with IPShell.dummy_mode = True, you
        can still have a specific call work by making it as IPShell(dummy=False).
        """
        self.keep_running = True
        clid = kw.pop('_call_location_id', None)
        if not clid:
            frame = sys._getframe(1)
            clid = '%s:%s' % (frame.f_code.co_filename, frame.f_lineno)
        self._call_location_id = clid
        if not self.embedded_active:
            return
        self.exit_now = False
        if dummy or (dummy != 0 and self.dummy_mode):
            return
        if header:
            self.old_banner2 = self.banner2
            self.banner2 = self.banner2 + '\n' + header + '\n'
        else:
            self.old_banner2 = ''
        if self.display_banner:
            self.show_banner()
        self.mainloop(local_ns, module, stack_depth=stack_depth, compile_flags=compile_flags)
        self.banner2 = self.old_banner2
        if self.exit_msg is not None:
            print(self.exit_msg)
        if self.should_raise:
            raise KillEmbedded('Embedded IPython raising error, as user requested.')

    def mainloop(self, local_ns=None, module=None, stack_depth=0, compile_flags=None):
        """Embeds IPython into a running python program.

        Parameters
        ----------
        local_ns, module
            Working local namespace (a dict) and module (a module or similar
            object). If given as None, they are automatically taken from the scope
            where the shell was called, so that program variables become visible.
        stack_depth : int
            How many levels in the stack to go to looking for namespaces (when
            local_ns or module is None). This allows an intermediate caller to
            make sure that this function gets the namespace from the intended
            level in the stack. By default (0) it will get its locals and globals
            from the immediate caller.
        compile_flags
            A bit field identifying the __future__ features
            that are enabled, as passed to the builtin :func:`compile` function.
            If given as None, they are automatically taken from the scope where
            the shell was called.

        """
        if (local_ns is None or module is None or compile_flags is None) and self.default_user_namespaces:
            call_frame = sys._getframe(stack_depth).f_back
            if local_ns is None:
                local_ns = call_frame.f_locals
            if module is None:
                global_ns = call_frame.f_globals
                try:
                    module = sys.modules[global_ns['__name__']]
                except KeyError:
                    warnings.warn('Failed to get module %s' % global_ns.get('__name__', 'unknown module'))
                    module = DummyMod()
                    module.__dict__ = global_ns
            if compile_flags is None:
                compile_flags = call_frame.f_code.co_flags & compilerop.PyCF_MASK
        orig_user_module = self.user_module
        orig_user_ns = self.user_ns
        orig_compile_flags = self.compile.flags
        if module is not None:
            self.user_module = module
        if local_ns is not None:
            reentrant_local_ns = {k: v for k, v in local_ns.items() if k not in self.user_ns_hidden.keys()}
            self.user_ns = reentrant_local_ns
            self.init_user_ns()
        if compile_flags is not None:
            self.compile.flags = compile_flags
        self.set_completer_frame()
        with self.builtin_trap, self.display_trap:
            self.interact()
        if local_ns is not None:
            local_ns.update({k: v for k, v in self.user_ns.items() if k not in self.user_ns_hidden.keys()})
        self.user_module = orig_user_module
        self.user_ns = orig_user_ns
        self.compile.flags = orig_compile_flags