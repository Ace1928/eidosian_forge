import difflib
import json
import os
import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import date
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, Union
import yaml
from ..models import auto as auto_module
from ..models.auto.configuration_auto import model_type_to_module_name
from ..utils import is_flax_available, is_tf_available, is_torch_available, logging
from . import BaseTransformersCLICommand
def add_content_to_file(file_name: Union[str, os.PathLike], content: str, add_after: Optional[Union[str, Pattern]]=None, add_before: Optional[Union[str, Pattern]]=None, exact_match: bool=False):
    """
    A utility to add some content inside a given file.

    Args:
       file_name (`str` or `os.PathLike`): The name of the file in which we want to insert some content.
       content (`str`): The content to add.
       add_after (`str` or `Pattern`):
           The pattern to test on a line of `text`, the new content is added after the first instance matching it.
       add_before (`str` or `Pattern`):
           The pattern to test on a line of `text`, the new content is added before the first instance matching it.
       exact_match (`bool`, *optional*, defaults to `False`):
           A line is considered a match with `add_after` or `add_before` if it matches exactly when `exact_match=True`,
           otherwise, if `add_after`/`add_before` is present in the line.

    <Tip warning={true}>

    The arguments `add_after` and `add_before` are mutually exclusive, and one exactly needs to be provided.

    </Tip>
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        old_content = f.read()
    new_content = add_content_to_text(old_content, content, add_after=add_after, add_before=add_before, exact_match=exact_match)
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(new_content)