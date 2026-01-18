import builtins as builtin_mod
import sys
import io as _io
import tokenize
from traitlets.config.configurable import Configurable
from traitlets import Instance, Float
from warnings import warn
class DisplayHook(Configurable):
    """The custom IPython displayhook to replace sys.displayhook.

    This class does many things, but the basic idea is that it is a callable
    that gets called anytime user code returns a value.
    """
    shell = Instance('IPython.core.interactiveshell.InteractiveShellABC', allow_none=True)
    exec_result = Instance('IPython.core.interactiveshell.ExecutionResult', allow_none=True)
    cull_fraction = Float(0.2)

    def __init__(self, shell=None, cache_size=1000, **kwargs):
        super(DisplayHook, self).__init__(shell=shell, **kwargs)
        cache_size_min = 3
        if cache_size <= 0:
            self.do_full_cache = 0
            cache_size = 0
        elif cache_size < cache_size_min:
            self.do_full_cache = 0
            cache_size = 0
            warn('caching was disabled (min value for cache size is %s).' % cache_size_min, stacklevel=3)
        else:
            self.do_full_cache = 1
        self.cache_size = cache_size
        self.shell = shell
        self._, self.__, self.___ = ('', '', '')
        to_user_ns = {'_': self._, '__': self.__, '___': self.___}
        self.shell.user_ns.update(to_user_ns)

    @property
    def prompt_count(self):
        return self.shell.execution_count

    def check_for_underscore(self):
        """Check if the user has set the '_' variable by hand."""
        if '_' in builtin_mod.__dict__:
            try:
                user_value = self.shell.user_ns['_']
                if user_value is not self._:
                    return
                del self.shell.user_ns['_']
            except KeyError:
                pass

    def quiet(self):
        """Should we silence the display hook because of ';'?"""
        try:
            cell = self.shell.history_manager.input_hist_parsed[-1]
        except IndexError:
            return False
        return self.semicolon_at_end_of_expression(cell)

    @staticmethod
    def semicolon_at_end_of_expression(expression):
        """Parse Python expression and detects whether last token is ';'"""
        sio = _io.StringIO(expression)
        tokens = list(tokenize.generate_tokens(sio.readline))
        for token in reversed(tokens):
            if token[0] in (tokenize.ENDMARKER, tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT):
                continue
            if token[0] == tokenize.OP and token[1] == ';':
                return True
            else:
                return False

    def start_displayhook(self):
        """Start the displayhook, initializing resources."""
        pass

    def write_output_prompt(self):
        """Write the output prompt.

        The default implementation simply writes the prompt to
        ``sys.stdout``.
        """
        sys.stdout.write(self.shell.separate_out)
        outprompt = 'Out[{}]: '.format(self.shell.execution_count)
        if self.do_full_cache:
            sys.stdout.write(outprompt)

    def compute_format_data(self, result):
        """Compute format data of the object to be displayed.

        The format data is a generalization of the :func:`repr` of an object.
        In the default implementation the format data is a :class:`dict` of
        key value pair where the keys are valid MIME types and the values
        are JSON'able data structure containing the raw data for that MIME
        type. It is up to frontends to determine pick a MIME to to use and
        display that data in an appropriate manner.

        This method only computes the format data for the object and should
        NOT actually print or write that to a stream.

        Parameters
        ----------
        result : object
            The Python object passed to the display hook, whose format will be
            computed.

        Returns
        -------
        (format_dict, md_dict) : dict
            format_dict is a :class:`dict` whose keys are valid MIME types and values are
            JSON'able raw data for that MIME type. It is recommended that
            all return values of this should always include the "text/plain"
            MIME type representation of the object.
            md_dict is a :class:`dict` with the same MIME type keys
            of metadata associated with each output.

        """
        return self.shell.display_formatter.format(result)
    prompt_end_newline = False

    def write_format_data(self, format_dict, md_dict=None) -> None:
        """Write the format data dict to the frontend.

        This default version of this method simply writes the plain text
        representation of the object to ``sys.stdout``. Subclasses should
        override this method to send the entire `format_dict` to the
        frontends.

        Parameters
        ----------
        format_dict : dict
            The format dict for the object passed to `sys.displayhook`.
        md_dict : dict (optional)
            The metadata dict to be associated with the display data.
        """
        if 'text/plain' not in format_dict:
            return
        result_repr = format_dict['text/plain']
        if '\n' in result_repr:
            if not self.prompt_end_newline:
                result_repr = '\n' + result_repr
        try:
            print(result_repr)
        except UnicodeEncodeError:
            print(result_repr.encode(sys.stdout.encoding, 'backslashreplace').decode(sys.stdout.encoding))

    def update_user_ns(self, result):
        """Update user_ns with various things like _, __, _1, etc."""
        if self.cache_size and result is not self.shell.user_ns['_oh']:
            if len(self.shell.user_ns['_oh']) >= self.cache_size and self.do_full_cache:
                self.cull_cache()
            update_unders = True
            for unders in ['_' * i for i in range(1, 4)]:
                if not unders in self.shell.user_ns:
                    continue
                if getattr(self, unders) is not self.shell.user_ns.get(unders):
                    update_unders = False
            self.___ = self.__
            self.__ = self._
            self._ = result
            if '_' not in builtin_mod.__dict__ and update_unders:
                self.shell.push({'_': self._, '__': self.__, '___': self.___}, interactive=False)
            to_main = {}
            if self.do_full_cache:
                new_result = '_%s' % self.prompt_count
                to_main[new_result] = result
                self.shell.push(to_main, interactive=False)
                self.shell.user_ns['_oh'][self.prompt_count] = result

    def fill_exec_result(self, result):
        if self.exec_result is not None:
            self.exec_result.result = result

    def log_output(self, format_dict):
        """Log the output."""
        if 'text/plain' not in format_dict:
            return
        if self.shell.logger.log_output:
            self.shell.logger.log_write(format_dict['text/plain'], 'output')
        self.shell.history_manager.output_hist_reprs[self.prompt_count] = format_dict['text/plain']

    def finish_displayhook(self):
        """Finish up all displayhook activities."""
        sys.stdout.write(self.shell.separate_out2)
        sys.stdout.flush()

    def __call__(self, result=None):
        """Printing with history cache management.

        This is invoked every time the interpreter needs to print, and is
        activated by setting the variable sys.displayhook to it.
        """
        self.check_for_underscore()
        if result is not None and (not self.quiet()):
            self.start_displayhook()
            self.write_output_prompt()
            format_dict, md_dict = self.compute_format_data(result)
            self.update_user_ns(result)
            self.fill_exec_result(result)
            if format_dict:
                self.write_format_data(format_dict, md_dict)
                self.log_output(format_dict)
            self.finish_displayhook()

    def cull_cache(self):
        """Output cache is full, cull the oldest entries"""
        oh = self.shell.user_ns.get('_oh', {})
        sz = len(oh)
        cull_count = max(int(sz * self.cull_fraction), 2)
        warn('Output cache limit (currently {sz} entries) hit.\nFlushing oldest {cull_count} entries.'.format(sz=sz, cull_count=cull_count))
        for i, n in enumerate(sorted(oh)):
            if i >= cull_count:
                break
            self.shell.user_ns.pop('_%i' % n, None)
            oh.pop(n, None)

    def flush(self):
        if not self.do_full_cache:
            raise ValueError("You shouldn't have reached the cache flush if full caching is not enabled!")
        for n in range(1, self.prompt_count + 1):
            key = '_' + repr(n)
            try:
                del self.shell.user_ns_hidden[key]
            except KeyError:
                pass
            try:
                del self.shell.user_ns[key]
            except KeyError:
                pass
        oh = self.shell.user_ns.get('_oh', None)
        if oh is not None:
            oh.clear()
        self._, self.__, self.___ = ('', '', '')
        if '_' not in builtin_mod.__dict__:
            self.shell.user_ns.update({'_': self._, '__': self.__, '___': self.___})
        import gc
        if sys.platform != 'cli':
            gc.collect()