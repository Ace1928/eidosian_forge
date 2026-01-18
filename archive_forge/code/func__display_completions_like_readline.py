from __future__ import unicode_literals
from prompt_toolkit.completion import CompleteEvent, get_common_complete_suffix
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.keys import Keys
from prompt_toolkit.key_binding.registry import Registry
import math
def _display_completions_like_readline(cli, completions):
    """
    Display the list of completions in columns above the prompt.
    This will ask for a confirmation if there are too many completions to fit
    on a single page and provide a paginator to walk through them.
    """
    from prompt_toolkit.shortcuts import create_confirm_application
    assert isinstance(completions, list)
    term_size = cli.output.get_size()
    term_width = term_size.columns
    term_height = term_size.rows
    max_compl_width = min(term_width, max((get_cwidth(c.text) for c in completions)) + 1)
    column_count = max(1, term_width // max_compl_width)
    completions_per_page = column_count * (term_height - 1)
    page_count = int(math.ceil(len(completions) / float(completions_per_page)))

    def display(page):
        page_completions = completions[page * completions_per_page:(page + 1) * completions_per_page]
        page_row_count = int(math.ceil(len(page_completions) / float(column_count)))
        page_columns = [page_completions[i * page_row_count:(i + 1) * page_row_count] for i in range(column_count)]
        result = []
        for r in range(page_row_count):
            for c in range(column_count):
                try:
                    result.append(page_columns[c][r].text.ljust(max_compl_width))
                except IndexError:
                    pass
            result.append('\n')
        cli.output.write(''.join(result))
        cli.output.flush()

    def run():
        if len(completions) > completions_per_page:
            message = 'Display all {} possibilities? (y on n) '.format(len(completions))
            confirm = (yield create_confirm_application(message))
            if confirm:
                for page in range(page_count):
                    display(page)
                    if page != page_count - 1:
                        show_more = (yield _create_more_application())
                        if not show_more:
                            return
            else:
                cli.output.write('\n')
                cli.output.flush()
        else:
            display(0)
    cli.run_application_generator(run, render_cli_done=True)