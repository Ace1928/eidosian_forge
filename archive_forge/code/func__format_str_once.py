import io
import json
import platform
import re
import sys
import tokenize
import traceback
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from enum import Enum
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import (
import click
from click.core import ParameterSource
from mypy_extensions import mypyc_attr
from pathspec import PathSpec
from pathspec.patterns.gitwildmatch import GitWildMatchPatternError
from _black_version import version as __version__
from black.cache import Cache
from black.comments import normalize_fmt_off
from black.const import (
from black.files import (
from black.handle_ipynb_magics import (
from black.linegen import LN, LineGenerator, transform_line
from black.lines import EmptyLineTracker, LinesBlock
from black.mode import FUTURE_FLAG_TO_FEATURE, VERSION_TO_FEATURES, Feature
from black.mode import Mode as Mode  # re-exported
from black.mode import Preview, TargetVersion, supports_feature
from black.nodes import (
from black.output import color_diff, diff, dump_to_file, err, ipynb_diff, out
from black.parsing import (  # noqa F401
from black.ranges import (
from black.report import Changed, NothingChanged, Report
from black.trans import iter_fexpr_spans
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def _format_str_once(src_contents: str, *, mode: Mode, lines: Collection[Tuple[int, int]]=()) -> str:
    src_node = lib2to3_parse(src_contents.lstrip(), mode.target_versions)
    dst_blocks: List[LinesBlock] = []
    if mode.target_versions:
        versions = mode.target_versions
    else:
        future_imports = get_future_imports(src_node)
        versions = detect_target_versions(src_node, future_imports=future_imports)
    context_manager_features = {feature for feature in {Feature.PARENTHESIZED_CONTEXT_MANAGERS} if supports_feature(versions, feature)}
    normalize_fmt_off(src_node, mode, lines)
    if lines:
        convert_unchanged_lines(src_node, lines)
    line_generator = LineGenerator(mode=mode, features=context_manager_features)
    elt = EmptyLineTracker(mode=mode)
    split_line_features = {feature for feature in {Feature.TRAILING_COMMA_IN_CALL, Feature.TRAILING_COMMA_IN_DEF} if supports_feature(versions, feature)}
    block: Optional[LinesBlock] = None
    for current_line in line_generator.visit(src_node):
        block = elt.maybe_empty_lines(current_line)
        dst_blocks.append(block)
        for line in transform_line(current_line, mode=mode, features=split_line_features):
            block.content_lines.append(str(line))
    if dst_blocks:
        dst_blocks[-1].after = 0
    dst_contents = []
    for block in dst_blocks:
        dst_contents.extend(block.all_lines())
    if not dst_contents:
        normalized_content, _, newline = decode_bytes(src_contents.encode('utf-8'))
        if '\n' in normalized_content:
            return newline
        return ''
    return ''.join(dst_contents)