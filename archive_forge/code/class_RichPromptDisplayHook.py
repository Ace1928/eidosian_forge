from pygments.token import Token
import sys
from IPython.core.displayhook import DisplayHook
from prompt_toolkit.formatted_text import fragment_list_width, PygmentsTokens
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.enums import EditingMode
class RichPromptDisplayHook(DisplayHook):
    """Subclass of base display hook using coloured prompt"""

    def write_output_prompt(self):
        sys.stdout.write(self.shell.separate_out)
        self.prompt_end_newline = True
        if self.do_full_cache:
            tokens = self.shell.prompts.out_prompt_tokens()
            prompt_txt = ''.join((s for _, s in tokens))
            if prompt_txt and (not prompt_txt.endswith('\n')):
                self.prompt_end_newline = False
            if self.shell.pt_app:
                print_formatted_text(PygmentsTokens(tokens), style=self.shell.pt_app.app.style, end='')
            else:
                sys.stdout.write(prompt_txt)

    def write_format_data(self, format_dict, md_dict=None) -> None:
        assert self.shell is not None
        if self.shell.mime_renderers:
            for mime, handler in self.shell.mime_renderers.items():
                if mime in format_dict:
                    handler(format_dict[mime], None)
                    return
        super().write_format_data(format_dict, md_dict)