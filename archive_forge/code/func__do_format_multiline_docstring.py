import argparse
import collections
import contextlib
import io
import re
import tokenize
from typing import TextIO, Tuple
import untokenize  # type: ignore
import docformatter.encode as _encode
import docformatter.strings as _strings
import docformatter.syntax as _syntax
import docformatter.util as _util
def _do_format_multiline_docstring(self, indentation: str, summary: str, description: str, open_quote: str) -> str:
    """Format multiline docstrings.

        Parameters
        ----------
        indentation : str
            The indentation to use for each line.
        summary : str
            The summary from the original docstring.
        description : str
            The long description from the original docstring.
        open_quote : str
            The type of quote used by the original docstring.  Selected from
            QUOTE_TYPES.

        Returns
        -------
        formatted_docstring : str
            The formatted docstring.
        """
    initial_indent = indentation if self.args.pre_summary_newline else 3 * ' ' + indentation
    pre_summary = '\n' + indentation if self.args.pre_summary_newline else ''
    summary = _syntax.wrap_summary(_strings.normalize_summary(summary, self.args.non_cap), wrap_length=self.args.wrap_summaries, initial_indent=initial_indent, subsequent_indent=indentation).lstrip()
    description = _syntax.wrap_description(description, indentation=indentation, wrap_length=self.args.wrap_descriptions, force_wrap=self.args.force_wrap, strict=self.args.non_strict, rest_sections=self.args.rest_section_adorns, style=self.args.style)
    post_description = '\n' if self.args.post_description_blank else ''
    return f'{open_quote}{pre_summary}{summary}\n\n{description}{post_description}\n{indentation}"""'