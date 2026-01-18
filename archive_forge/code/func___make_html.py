import itertools
import os
import re
from string import Template
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple
from tokenizers import Encoding, Tokenizer
@staticmethod
def __make_html(text: str, encoding: Encoding, annotations: AnnotationList) -> str:
    char_states = EncodingVisualizer.__make_char_states(text, encoding, annotations)
    current_consecutive_chars = [char_states[0]]
    prev_anno_ix = char_states[0].anno_ix
    spans = []
    label_colors_dict = EncodingVisualizer.calculate_label_colors(annotations)
    cur_anno_ix = char_states[0].anno_ix
    if cur_anno_ix is not None:
        anno = annotations[cur_anno_ix]
        label = anno.label
        color = label_colors_dict[label]
        spans.append(f'<span class="annotation" style="color:{color}" data-label="{label}">')
    for cs in char_states[1:]:
        cur_anno_ix = cs.anno_ix
        if cur_anno_ix != prev_anno_ix:
            spans.append(EncodingVisualizer.consecutive_chars_to_html(current_consecutive_chars, text=text, encoding=encoding))
            current_consecutive_chars = [cs]
            if prev_anno_ix is not None:
                spans.append('</span>')
            if cur_anno_ix is not None:
                anno = annotations[cur_anno_ix]
                label = anno.label
                color = label_colors_dict[label]
                spans.append(f'<span class="annotation" style="color:{color}" data-label="{label}">')
        prev_anno_ix = cur_anno_ix
        if cs.partition_key() == current_consecutive_chars[0].partition_key():
            current_consecutive_chars.append(cs)
        else:
            spans.append(EncodingVisualizer.consecutive_chars_to_html(current_consecutive_chars, text=text, encoding=encoding))
            current_consecutive_chars = [cs]
    spans.append(EncodingVisualizer.consecutive_chars_to_html(current_consecutive_chars, text=text, encoding=encoding))
    res = HTMLBody(spans)
    return res