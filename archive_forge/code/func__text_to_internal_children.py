import os
from datetime import datetime
from typing import Dict, Iterable, Optional, Tuple, Union
from typing import List as LList
from urllib.parse import urlparse, urlunparse
from pydantic import ConfigDict, Field, validator
from pydantic.dataclasses import dataclass
import wandb
from . import expr_parsing, gql, internal
from .internal import (
def _text_to_internal_children(text_field):
    text = text_field
    if text == []:
        text = ''
    if not isinstance(text, list):
        text = [text]
    texts = []
    for x in text:
        t = None
        if isinstance(x, str):
            t = internal.Text(text=x)
        elif isinstance(x, TextWithInlineComments):
            t = internal.Text(text=x.text, inline_comments=x._inline_comments)
        elif isinstance(x, Link):
            txt = x.text
            if isinstance(txt, str):
                children = [internal.Text(text=txt)]
            elif isinstance(txt, TextWithInlineComments):
                children = [internal.Text(text=txt.text, inline_comments=txt._inline_comments)]
            t = internal.InlineLink(url=x.url, children=children)
        elif isinstance(x, InlineLatex):
            t = internal.InlineLatex(content=x.text)
        elif isinstance(x, InlineCode):
            t = internal.Text(text=x.text, inline_code=True)
        texts.append(t)
    if not all((isinstance(x, str) for x in texts)):
        pass
    return texts