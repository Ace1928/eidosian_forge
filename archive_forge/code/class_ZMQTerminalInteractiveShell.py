from __future__ import print_function
import asyncio
import base64
import errno
from getpass import getpass
from io import BytesIO
import os
from queue import Empty
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from warnings import warn
from typing import Dict as DictType, Any as AnyType
from zmq import ZMQError
from IPython.core import page
from traitlets import (
from traitlets.config import SingletonConfigurable
from .completer import ZMQCompleter
from .zmqhistory import ZMQHistoryManager
from . import __version__
from prompt_toolkit import __version__ as ptk_version
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.enums import DEFAULT_BUFFER, EditingMode
from prompt_toolkit.filters import (
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.shortcuts.prompt import PromptSession
from prompt_toolkit.shortcuts import print_formatted_text, CompleteStyle
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.layout.processors import (
from prompt_toolkit.styles import merge_styles
from prompt_toolkit.styles.pygments import (style_from_pygments_cls,
from prompt_toolkit.formatted_text import PygmentsTokens
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.utils import suspend_to_background_supported
from pygments.styles import get_style_by_name
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound
from pygments.token import Token
from jupyter_console.utils import run_sync, ensure_async
class ZMQTerminalInteractiveShell(SingletonConfigurable):
    readline_use = False
    pt_cli = None
    _executing = False
    _execution_state = Unicode('')
    _pending_clearoutput = False
    _eventloop = None
    own_kernel = False
    editing_mode = Unicode('emacs', config=True, help="Shortcut style to use at the prompt. 'vi' or 'emacs'.")
    highlighting_style = Unicode('', config=True, help='The name of a Pygments style to use for syntax highlighting')
    highlighting_style_overrides = Dict(config=True, help='Override highlighting format for specific tokens')
    true_color = Bool(False, config=True, help='Use 24bit colors instead of 256 colors in prompt highlighting. If your terminal supports true color, the following command should print \'TRUECOLOR\' in orange: printf "\\x1b[38;2;255;100;0mTRUECOLOR\\x1b[0m\\n"')
    history_load_length = Integer(1000, config=True, help='How many history items to load into memory')
    banner = Unicode('Jupyter console {version}\n\n{kernel_banner}', config=True, help='Text to display before the first prompt. Will be formatted with variables {version} and {kernel_banner}.')
    kernel_timeout = Float(60, config=True, help='Timeout for giving up on a kernel (in seconds).\n\n        On first connect and restart, the console tests whether the\n        kernel is running and responsive by sending kernel_info_requests.\n        This sets the timeout in seconds for how long the kernel can take\n        before being presumed dead.\n        ')
    image_handler = Enum(('PIL', 'stream', 'tempfile', 'callable'), 'PIL', config=True, allow_none=True, help="\n        Handler for image type output.  This is useful, for example,\n        when connecting to the kernel in which pylab inline backend is\n        activated.  There are four handlers defined.  'PIL': Use\n        Python Imaging Library to popup image; 'stream': Use an\n        external program to show the image.  Image will be fed into\n        the STDIN of the program.  You will need to configure\n        `stream_image_handler`; 'tempfile': Use an external program to\n        show the image.  Image will be saved in a temporally file and\n        the program is called with the temporally file.  You will need\n        to configure `tempfile_image_handler`; 'callable': You can set\n        any Python callable which is called with the image data.  You\n        will need to configure `callable_image_handler`.\n        ")
    stream_image_handler = List(config=True, help="\n        Command to invoke an image viewer program when you are using\n        'stream' image handler.  This option is a list of string where\n        the first element is the command itself and reminders are the\n        options for the command.  Raw image data is given as STDIN to\n        the program.\n        ")
    tempfile_image_handler = List(config=True, help="\n        Command to invoke an image viewer program when you are using\n        'tempfile' image handler.  This option is a list of string\n        where the first element is the command itself and reminders\n        are the options for the command.  You can use {file} and\n        {format} in the string to represent the location of the\n        generated image file and image format.\n        ")
    callable_image_handler = Any(config=True, help='\n        Callable object called via \'callable\' image handler with one\n        argument, `data`, which is `msg["content"]["data"]` where\n        `msg` is the message from iopub channel.  For example, you can\n        find base64 encoded PNG data as `data[\'image/png\']`. If your function\n        can\'t handle the data supplied, it should return `False` to indicate\n        this.\n        ')
    mime_preference = List(default_value=['image/png', 'image/jpeg', 'image/svg+xml'], config=True, help='\n        Preferred object representation MIME type in order.  First\n        matched MIME type will be used.\n        ')
    use_kernel_is_complete = Bool(True, config=True, help="Whether to use the kernel's is_complete message\n        handling. If False, then the frontend will use its\n        own is_complete handler.\n        ")
    kernel_is_complete_timeout = Float(1, config=True, help="Timeout (in seconds) for giving up on a kernel's is_complete\n        response.\n\n        If the kernel does not respond at any point within this time,\n        the kernel will no longer be asked if code is complete, and the\n        console will default to the built-in is_complete test.\n        ")
    confirm_exit = Bool(True, help="Set to display confirmation dialog on exit.\n        You can always use 'exit' or 'quit', to force a\n        direct exit without any confirmation.\n        ")
    display_completions = Enum(('column', 'multicolumn', 'readlinelike'), help="Options for displaying tab completions, 'column', 'multicolumn', and 'readlinelike'. These options are for `prompt_toolkit`, see `prompt_toolkit` documentation for more information.", default_value='multicolumn').tag(config=True)
    prompt_includes_vi_mode = Bool(True, help='Display the current vi mode (when using vi editing mode).').tag(config=True)
    highlight_matching_brackets = Bool(True, help='Highlight matching brackets.').tag(config=True)
    manager = Instance('jupyter_client.KernelManager', allow_none=True)
    client = Instance('jupyter_client.KernelClient', allow_none=True)

    def _client_changed(self, name, old, new):
        self.session_id = new.session.session
    session_id = Unicode()

    def _banner1_default(self):
        return 'Jupyter Console {version}\n'.format(version=__version__)
    simple_prompt = Bool(False, help='Use simple fallback prompt. Features may be limited.').tag(config=True)

    def __init__(self, **kwargs):
        super(ZMQTerminalInteractiveShell, self).__init__(**kwargs)
        self.configurables = [self]
        self.init_history()
        self.init_completer()
        self.init_io()
        self.init_kernel_info()
        self.init_prompt_toolkit_cli()
        self.keep_running = True
        self.execution_count = 1

    def init_completer(self):
        """Initialize the completion machinery.

        This creates completion machinery that can be used by client code,
        either interactively in-process (typically triggered by the readline
        library), programmatically (such as in test suites) or out-of-process
        (typically over the network by remote frontends).
        """
        self.Completer = ZMQCompleter(self, self.client, config=self.config)

    def init_history(self):
        """Sets up the command history. """
        self.history_manager = ZMQHistoryManager(client=self.client)
        self.configurables.append(self.history_manager)

    def vi_mode(self):
        if getattr(self, 'editing_mode', None) == 'vi' and self.prompt_includes_vi_mode:
            return '[' + str(self.pt_cli.app.vi_state.input_mode)[3:6] + '] '
        return ''

    def get_prompt_tokens(self, ec=None):
        if ec is None:
            ec = self.execution_count
        return [(Token.Prompt, self.vi_mode()), (Token.Prompt, 'In ['), (Token.PromptNum, str(ec)), (Token.Prompt, ']: ')]

    def get_continuation_tokens(self, width):
        return [(Token.Prompt, ' ' * (width - 5) + '...: ')]

    def get_out_prompt_tokens(self):
        return [(Token.OutPrompt, 'Out['), (Token.OutPromptNum, str(self.execution_count)), (Token.OutPrompt, ']: ')]

    def print_out_prompt(self):
        tokens = self.get_out_prompt_tokens()
        print_formatted_text(PygmentsTokens(tokens), end='', style=self.pt_cli.app.style)

    def get_remote_prompt_tokens(self):
        return [(Token.RemotePrompt, self.other_output_prefix)]

    def print_remote_prompt(self, ec=None):
        tokens = self.get_remote_prompt_tokens() + self.get_prompt_tokens(ec=ec)
        print_formatted_text(PygmentsTokens(tokens), end='', style=self.pt_cli.app.style)

    @property
    def pt_complete_style(self):
        return {'multicolumn': CompleteStyle.MULTI_COLUMN, 'column': CompleteStyle.COLUMN, 'readlinelike': CompleteStyle.READLINE_LIKE}[self.display_completions]
    kernel_info: DictType[str, AnyType] = {}

    def init_kernel_info(self):
        """Wait for a kernel to be ready, and store kernel info"""
        timeout = self.kernel_timeout
        tic = time.time()
        self.client.hb_channel.unpause()
        msg_id = self.client.kernel_info()
        while True:
            try:
                reply = self.client.get_shell_msg(timeout=1)
            except Empty as e:
                if time.time() - tic > timeout:
                    raise RuntimeError("Kernel didn't respond to kernel_info_request") from e
            else:
                if reply['parent_header'].get('msg_id') == msg_id:
                    self.kernel_info = reply['content']
                    return

    def show_banner(self):
        print(self.banner.format(version=__version__, kernel_banner=self.kernel_info.get('banner', '')), end='', flush=True)

    def init_prompt_toolkit_cli(self):
        if self.simple_prompt or 'JUPYTER_CONSOLE_TEST' in os.environ:

            async def prompt():
                prompt = 'In [%d]: ' % self.execution_count
                raw = await async_input(prompt)
                return raw
            self.prompt_for_code = prompt
            self.print_out_prompt = lambda: print('Out[%d]: ' % self.execution_count, end='')
            return
        kb = KeyBindings()
        insert_mode = vi_insert_mode | emacs_insert_mode

        @kb.add('enter', filter=has_focus(DEFAULT_BUFFER) & ~has_selection & insert_mode)
        def _(event):
            b = event.current_buffer
            d = b.document
            if not (d.on_last_line or d.cursor_position_row >= d.line_count - d.empty_line_count_at_the_end()):
                b.newline()
                return
            self.handle_iopub()
            more, indent = self.check_complete(d.text)
            if not more and b.accept_handler:
                b.validate_and_handle()
            else:
                b.insert_text('\n' + indent)

        @kb.add('c-c', filter=has_focus(DEFAULT_BUFFER))
        def _(event):
            event.current_buffer.reset()

        @kb.add('c-\\', filter=has_focus(DEFAULT_BUFFER))
        def _(event):
            raise EOFError

        @kb.add('c-z', filter=Condition(lambda: suspend_to_background_supported()))
        def _(event):
            event.cli.suspend_to_background()

        @kb.add('c-o', filter=has_focus(DEFAULT_BUFFER) & emacs_insert_mode)
        def _(event):
            event.current_buffer.insert_text('\n')
        history = InMemoryHistory()
        last_cell = u''
        for _, _, cell in self.history_manager.get_tail(self.history_load_length, include_latest=True):
            cell = cell.rstrip()
            if cell and cell != last_cell:
                history.append_string(cell)
        style_overrides = {Token.Prompt: '#009900', Token.PromptNum: '#00ff00 bold', Token.OutPrompt: '#ff2200', Token.OutPromptNum: '#ff0000 bold', Token.RemotePrompt: '#999900'}
        if self.highlighting_style:
            style_cls = get_style_by_name(self.highlighting_style)
        else:
            style_cls = get_style_by_name('default')
            style_overrides.update({Token.Number: '#007700', Token.Operator: 'noinherit', Token.String: '#BB6622', Token.Name.Function: '#2080D0', Token.Name.Class: 'bold #2080D0', Token.Name.Namespace: 'bold #2080D0'})
        style_overrides.update(self.highlighting_style_overrides)
        style = merge_styles([style_from_pygments_cls(style_cls), style_from_pygments_dict(style_overrides)])
        editing_mode = getattr(EditingMode, self.editing_mode.upper())
        langinfo = self.kernel_info.get('language_info', {})
        lexer = langinfo.get('pygments_lexer', langinfo.get('name', 'text'))
        input_processors = [ConditionalProcessor(processor=HighlightMatchingBracketProcessor(chars='[](){}'), filter=has_focus(DEFAULT_BUFFER) & ~is_done & Condition(lambda: self.highlight_matching_brackets))]
        if not PTK3:
            use_asyncio_event_loop()
        self.pt_cli = PromptSession(message=lambda: PygmentsTokens(self.get_prompt_tokens()), multiline=True, complete_style=self.pt_complete_style, editing_mode=editing_mode, lexer=PygmentsLexer(get_pygments_lexer(lexer)), prompt_continuation=lambda width, lineno, is_soft_wrap: PygmentsTokens(self.get_continuation_tokens(width)), key_bindings=kb, history=history, completer=JupyterPTCompleter(self.Completer), enable_history_search=True, style=style, input_processors=input_processors, color_depth=ColorDepth.TRUE_COLOR if self.true_color else None)

    async def prompt_for_code(self):
        if self.next_input:
            default = self.next_input
            self.next_input = None
        else:
            default = ''
        if PTK3:
            text = await self.pt_cli.prompt_async(default=default)
        else:
            text = await self.pt_cli.prompt(default=default, async_=True)
        return text

    def init_io(self):
        if sys.platform not in {'win32', 'cli'}:
            return
        import colorama
        colorama.init()

    def check_complete(self, code):
        if self.use_kernel_is_complete:
            msg_id = self.client.is_complete(code)
            try:
                return self.handle_is_complete_reply(msg_id, timeout=self.kernel_is_complete_timeout)
            except SyntaxError:
                return (False, '')
        else:
            lines = code.splitlines()
            if len(lines):
                more = lines[-1] != ''
                return (more, '')
            else:
                return (False, '')

    def ask_exit(self):
        self.keep_running = False
    next_input = None

    def pre_prompt(self):
        if self.next_input:
            s = self.next_input

            def set_doc():
                self.pt_cli.app.buffer.document = Document(s)
            if hasattr(self.pt_cli, 'pre_run_callables'):
                self.pt_cli.app.pre_run_callables.append(set_doc)
            else:
                set_doc()
            self.next_input = None

    async def interact(self, loop=None, display_banner=None):
        while self.keep_running:
            print('\n', end='')
            try:
                code = await self.prompt_for_code()
            except EOFError:
                if not self.confirm_exit or ask_yes_no('Do you really want to exit ([y]/n)?', 'y', 'n'):
                    self.ask_exit()
            else:
                if code:
                    self.run_cell(code, store_history=True)

    async def _main_task(self):
        loop = asyncio.get_running_loop()
        tasks = [asyncio.create_task(self.interact(loop=loop))]
        if self.include_other_output:
            tasks.append(asyncio.create_task(self.handle_external_iopub(loop=loop)))
        _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        try:
            await asyncio.gather(*pending)
        except asyncio.CancelledError:
            pass

    def mainloop(self):
        self.keepkernel = not self.own_kernel
        while True:
            try:
                asyncio.run(self._main_task())
                break
            except KeyboardInterrupt:
                print('\nKeyboardInterrupt escaped interact()\n')
        if self._eventloop:
            self._eventloop.close()
        if self.keepkernel and (not self.own_kernel):
            print('keeping kernel alive')
        elif self.keepkernel and self.own_kernel:
            print('owning kernel, cannot keep it alive')
            self.client.shutdown()
        else:
            print('Shutting down kernel')
            self.client.shutdown()

    def run_cell(self, cell, store_history=True):
        """Run a complete IPython cell.

        Parameters
        ----------
        cell : str
          The code (including IPython code such as %magic functions) to run.
        store_history : bool
          If True, the raw and translated cell will be stored in IPython's
          history. For user code calling back into IPython's machinery, this
          should be set to False.
        """
        if not cell or cell.isspace():
            self.handle_iopub()
            return
        while run_sync(self.client.shell_channel.msg_ready)():
            run_sync(self.client.shell_channel.get_msg)()
        msg_id = self.client.execute(cell, not store_history)
        self._executing = True
        self._execution_state = 'busy'
        while self._execution_state != 'idle' and self.client.is_alive():
            try:
                self.handle_input_request(msg_id, timeout=0.05)
            except Empty:
                self.handle_iopub(msg_id)
            except ZMQError as e:
                if e.errno != errno.EINTR:
                    raise
        while self.client.is_alive():
            try:
                self.handle_execute_reply(msg_id, timeout=0.05)
            except Empty:
                pass
            else:
                break
        self._executing = False

    def handle_execute_reply(self, msg_id, timeout=None):
        kwargs = {'timeout': timeout}
        msg = run_sync(self.client.shell_channel.get_msg)(**kwargs)
        if msg['parent_header'].get('msg_id', None) == msg_id:
            self.handle_iopub(msg_id)
            content = msg['content']
            status = content['status']
            if status == 'aborted':
                sys.stdout.write('Aborted\n')
                return
            elif status == 'ok':
                for item in content.get('payload', []):
                    source = item['source']
                    if source == 'page':
                        page.page(item['data']['text/plain'])
                    elif source == 'set_next_input':
                        self.next_input = item['text']
                    elif source == 'ask_exit':
                        self.keepkernel = item.get('keepkernel', False)
                        self.ask_exit()
            elif status == 'error':
                pass
            self.execution_count = int(content['execution_count'] + 1)

    def handle_is_complete_reply(self, msg_id, timeout=None):
        """
        Wait for a repsonse from the kernel, and return two values:
            more? - (boolean) should the frontend ask for more input
            indent - an indent string to prefix the input
        Overloaded methods may want to examine the comeplete source. Its is
        in the self._source_lines_buffered list.
        """
        msg = None
        try:
            kwargs = {'timeout': timeout}
            msg = run_sync(self.client.shell_channel.get_msg)(**kwargs)
        except Empty:
            warn('The kernel did not respond to an is_complete_request. Setting `use_kernel_is_complete` to False.')
            self.use_kernel_is_complete = False
            return (False, '')
        if msg['parent_header'].get('msg_id', None) != msg_id:
            warn('The kernel did not respond properly to an is_complete_request: %s.' % str(msg))
            return (False, '')
        else:
            status = msg['content'].get('status', None)
            indent = msg['content'].get('indent', '')
        if status == 'complete':
            return (False, indent)
        elif status == 'incomplete':
            return (True, indent)
        elif status == 'invalid':
            raise SyntaxError()
        elif status == 'unknown':
            return (False, indent)
        else:
            warn('The kernel sent an invalid is_complete_reply status: "%s".' % status)
            return (False, indent)
    include_other_output = Bool(False, config=True, help='Whether to include output from clients\n        other than this one sharing the same kernel.\n        ')
    other_output_prefix = Unicode('Remote ', config=True, help='Prefix to add to outputs coming from clients other than this one.\n\n        Only relevant if include_other_output is True.\n        ')

    def from_here(self, msg):
        """Return whether a message is from this session"""
        return msg['parent_header'].get('session', self.session_id) == self.session_id

    def include_output(self, msg):
        """Return whether we should include a given output message"""
        from_here = self.from_here(msg)
        if msg['msg_type'] == 'execute_input':
            return self.include_other_output and (not from_here)
        if self.include_other_output:
            return True
        else:
            return from_here

    async def handle_external_iopub(self, loop=None):
        while self.keep_running:
            poll_result = await ensure_async(self.client.iopub_channel.socket.poll(0))
            if poll_result:
                self.handle_iopub()
            await asyncio.sleep(0.5)

    def handle_iopub(self, msg_id=''):
        """Process messages on the IOPub channel

           This method consumes and processes messages on the IOPub channel,
           such as stdout, stderr, execute_result and status.

           It only displays output that is caused by this session.
        """
        while run_sync(self.client.iopub_channel.msg_ready)():
            sub_msg = run_sync(self.client.iopub_channel.get_msg)()
            msg_type = sub_msg['header']['msg_type']
            if msg_type == 'execute_input':
                self.execution_count = int(sub_msg['content']['execution_count']) + 1
            if self.include_output(sub_msg):
                if msg_type == 'status':
                    self._execution_state = sub_msg['content']['execution_state']
                elif msg_type == 'stream':
                    if sub_msg['content']['name'] == 'stdout':
                        if self._pending_clearoutput:
                            print('\r', end='')
                            self._pending_clearoutput = False
                        print(sub_msg['content']['text'], end='')
                        sys.stdout.flush()
                    elif sub_msg['content']['name'] == 'stderr':
                        if self._pending_clearoutput:
                            print('\r', file=sys.stderr, end='')
                            self._pending_clearoutput = False
                        print(sub_msg['content']['text'], file=sys.stderr, end='')
                        sys.stderr.flush()
                elif msg_type == 'execute_result':
                    if self._pending_clearoutput:
                        print('\r', end='')
                        self._pending_clearoutput = False
                    self.execution_count = int(sub_msg['content']['execution_count'])
                    if not self.from_here(sub_msg):
                        sys.stdout.write(self.other_output_prefix)
                    format_dict = sub_msg['content']['data']
                    self.handle_rich_data(format_dict)
                    if 'text/plain' not in format_dict:
                        continue
                    sys.stdout.flush()
                    sys.stderr.flush()
                    self.print_out_prompt()
                    text_repr = format_dict['text/plain']
                    if '\n' in text_repr:
                        print()
                    print(text_repr)
                    if not self.from_here(sub_msg):
                        sys.stdout.write('\n')
                        sys.stdout.flush()
                        self.print_remote_prompt()
                elif msg_type == 'display_data':
                    data = sub_msg['content']['data']
                    handled = self.handle_rich_data(data)
                    if not handled:
                        if not self.from_here(sub_msg):
                            sys.stdout.write(self.other_output_prefix)
                        if 'text/plain' in data:
                            print(data['text/plain'])
                elif msg_type == 'execute_input':
                    content = sub_msg['content']
                    ec = content.get('execution_count', self.execution_count - 1)
                    sys.stdout.write('\n')
                    sys.stdout.flush()
                    self.print_remote_prompt(ec=ec)
                    sys.stdout.write(content['code'] + '\n')
                elif msg_type == 'clear_output':
                    if sub_msg['content']['wait']:
                        self._pending_clearoutput = True
                    else:
                        print('\r', end='')
                elif msg_type == 'error':
                    for frame in sub_msg['content']['traceback']:
                        print(frame, file=sys.stderr)
    _imagemime = {'image/png': 'png', 'image/jpeg': 'jpeg', 'image/svg+xml': 'svg'}

    def handle_rich_data(self, data):
        for mime in self.mime_preference:
            if mime in data and mime in self._imagemime:
                if self.handle_image(data, mime):
                    return True
        return False

    def handle_image(self, data, mime):
        handler = getattr(self, 'handle_image_{0}'.format(self.image_handler), None)
        if handler:
            return handler(data, mime)

    def handle_image_PIL(self, data, mime):
        if mime not in ('image/png', 'image/jpeg'):
            return False
        try:
            from PIL import Image, ImageShow
        except ImportError:
            return False
        raw = base64.decodebytes(data[mime].encode('ascii'))
        img = Image.open(BytesIO(raw))
        return ImageShow.show(img)

    def handle_image_stream(self, data, mime):
        raw = base64.decodebytes(data[mime].encode('ascii'))
        imageformat = self._imagemime[mime]
        fmt = dict(format=imageformat)
        args = [s.format(**fmt) for s in self.stream_image_handler]
        with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as proc:
            proc.communicate(raw)
            return proc.returncode == 0

    def handle_image_tempfile(self, data, mime):
        raw = base64.decodebytes(data[mime].encode('ascii'))
        imageformat = self._imagemime[mime]
        filename = 'tmp.{0}'.format(imageformat)
        with TemporaryDirectory() as tempdir:
            fullpath = os.path.join(tempdir, filename)
            with open(fullpath, 'wb') as f:
                f.write(raw)
            fmt = dict(file=fullpath, format=imageformat)
            args = [s.format(**fmt) for s in self.tempfile_image_handler]
            rc = subprocess.call(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return rc == 0

    def handle_image_callable(self, data, mime):
        res = self.callable_image_handler(data)
        if res is not False:
            res = True
        return res

    def handle_input_request(self, msg_id, timeout=0.1):
        """ Method to capture raw_input
        """
        req = run_sync(self.client.stdin_channel.get_msg)(timeout=timeout)
        self.handle_iopub(msg_id)
        if msg_id == req['parent_header'].get('msg_id'):
            real_handler = signal.getsignal(signal.SIGINT)

            def double_int(sig, frame):
                real_handler(sig, frame)
                raise KeyboardInterrupt
            signal.signal(signal.SIGINT, double_int)
            content = req['content']
            read = getpass if content.get('password', False) else input
            try:
                raw_data = read(content['prompt'])
            except EOFError:
                raw_data = '\x04'
            except KeyboardInterrupt:
                sys.stdout.write('\n')
                return
            finally:
                signal.signal(signal.SIGINT, real_handler)
            if not (run_sync(self.client.stdin_channel.msg_ready)() or run_sync(self.client.shell_channel.msg_ready)()):
                self.client.input(raw_data)