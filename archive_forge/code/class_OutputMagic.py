import sys
import time
from IPython.core.magic import Magics, line_cell_magic, line_magic, magics_class
from IPython.display import HTML, display
from ..core.options import Options, Store, StoreOptions, options_policy
from ..core.pprint import InfoPrinter
from ..operation import Compositor
from IPython.core import page
@magics_class
class OutputMagic(Magics):

    @classmethod
    def info(cls, obj):
        disabled = Store.output_settings._disable_info_output
        if Store.output_settings.options['info'] and (not disabled):
            page.page(InfoPrinter.info(obj, ansi=True))

    @classmethod
    def pprint(cls):
        """
        Pretty print the current element options
        """
        current, count = ('', 0)
        for k, v in Store.output_settings.options.items():
            keyword = f'{k}={v!r}'
            if len(current) + len(keyword) > 80:
                print(('%output' if count == 0 else '      ') + current)
                count += 1
                current = keyword
            else:
                current += ' ' + keyword
        else:
            print(('%output' if count == 0 else '      ') + current)

    @classmethod
    def option_completer(cls, k, v):
        raw_line = v.text_until_cursor
        line = raw_line.replace('%output', '')
        completion_key = None
        tokens = [t for els in reversed(line.split('=')) for t in els.split()]
        for token in tokens:
            if token.strip() in Store.output_settings.allowed:
                completion_key = token.strip()
                break
        values = [val for val in Store.output_settings.allowed.get(completion_key, []) if val not in Store.output_settings.hidden.get(completion_key, [])]
        vreprs = [repr(el) for el in values if not isinstance(el, tuple)]
        return vreprs + [el + '=' for el in Store.output_settings.allowed.keys()]

    @line_cell_magic
    def output(self, line, cell=None):
        if line == '':
            self.pprint()
            print('\nFor help with the %output magic, call %output?')
            return

        def cell_runner(cell, renderer):
            self.shell.run_cell(cell, store_history=STORE_HISTORY)

        def warnfn(msg):
            display(HTML(f'<b>Warning:</b> {msg}'))
        if line:
            help_prompt = 'For help with the %output magic, call %output?\n'
        else:
            help_prompt = 'For help with the %%output magic, call %%output?\n'
        Store.output_settings.output(line, cell, cell_runner=cell_runner, help_prompt=help_prompt, warnfn=warnfn)