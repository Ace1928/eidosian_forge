import itertools
import os
import re
from string import Template
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple
from tokenizers import Encoding, Tokenizer
class EncodingVisualizer:
    """
    Build an EncodingVisualizer

    Args:

         tokenizer (:class:`~tokenizers.Tokenizer`):
            A tokenizer instance

         default_to_notebook (:obj:`bool`):
            Whether to render html output in a notebook by default

         annotation_converter (:obj:`Callable`, `optional`):
            An optional (lambda) function that takes an annotation in any format and returns
            an Annotation object
    """
    unk_token_regex = re.compile('(.{1}\x08)?(unk|oov)(\x08.{1})?', flags=re.IGNORECASE)

    def __init__(self, tokenizer: Tokenizer, default_to_notebook: bool=True, annotation_converter: Optional[Callable[[Any], Annotation]]=None):
        if default_to_notebook:
            try:
                from IPython.core.display import HTML, display
            except ImportError as e:
                raise Exception("We couldn't import IPython utils for html display.\n                        Are you running in a notebook?\n                        You can also pass `default_to_notebook=False` to get back raw HTML\n                    ")
        self.tokenizer = tokenizer
        self.default_to_notebook = default_to_notebook
        self.annotation_coverter = annotation_converter
        pass

    def __call__(self, text: str, annotations: AnnotationList=[], default_to_notebook: Optional[bool]=None) -> Optional[str]:
        """
        Build a visualization of the given text

        Args:
            text (:obj:`str`):
                The text to tokenize

            annotations (:obj:`List[Annotation]`, `optional`):
                An optional list of annotations of the text. The can either be an annotation class
                or anything else if you instantiated the visualizer with a converter function

            default_to_notebook (:obj:`bool`, `optional`, defaults to `False`):
                If True, will render the html in a notebook. Otherwise returns an html string.

        Returns:
            The HTML string if default_to_notebook is False, otherwise (default) returns None and
            renders the HTML in the notebook

        """
        final_default_to_notebook = self.default_to_notebook
        if default_to_notebook is not None:
            final_default_to_notebook = default_to_notebook
        if final_default_to_notebook:
            try:
                from IPython.core.display import HTML, display
            except ImportError as e:
                raise Exception("We couldn't import IPython utils for html display.\n                    Are you running in a notebook?")
        if self.annotation_coverter is not None:
            annotations = list(map(self.annotation_coverter, annotations))
        encoding = self.tokenizer.encode(text)
        html = EncodingVisualizer.__make_html(text, encoding, annotations)
        if final_default_to_notebook:
            display(HTML(html))
        else:
            return html

    @staticmethod
    def calculate_label_colors(annotations: AnnotationList) -> Dict[str, str]:
        """
        Generates a color palette for all the labels in a given set of annotations

        Args:
          annotations (:obj:`Annotation`):
            A list of annotations

        Returns:
            :obj:`dict`: A dictionary mapping labels to colors in HSL format
        """
        if len(annotations) == 0:
            return {}
        labels = set(map(lambda x: x.label, annotations))
        num_labels = len(labels)
        h_step = int(255 / num_labels)
        if h_step < 20:
            h_step = 20
        s = 32
        l = 64
        h = 10
        colors = {}
        for label in sorted(labels):
            colors[label] = f'hsl({h},{s}%,{l}%'
            h += h_step
        return colors

    @staticmethod
    def consecutive_chars_to_html(consecutive_chars_list: List[CharState], text: str, encoding: Encoding):
        """
        Converts a list of "consecutive chars" into a single HTML element.
        Chars are consecutive if they fall under the same word, token and annotation.
        The CharState class is a named tuple with a "partition_key" method that makes it easy to
        compare if two chars are consecutive.

        Args:
            consecutive_chars_list (:obj:`List[CharState]`):
                A list of CharStates that have been grouped together

            text (:obj:`str`):
                The original text being processed

            encoding (:class:`~tokenizers.Encoding`):
                The encoding returned from the tokenizer

        Returns:
            :obj:`str`: The HTML span for a set of consecutive chars
        """
        first = consecutive_chars_list[0]
        if first.char_ix is None:
            stoken = encoding.tokens[first.token_ix]
            return f'<span class="special-token" data-stoken={stoken}></span>'
        last = consecutive_chars_list[-1]
        start = first.char_ix
        end = last.char_ix + 1
        span_text = text[start:end]
        css_classes = []
        data_items = {}
        if first.token_ix is not None:
            css_classes.append('token')
            if first.is_multitoken:
                css_classes.append('multi-token')
            if first.token_ix % 2:
                css_classes.append('odd-token')
            else:
                css_classes.append('even-token')
            if EncodingVisualizer.unk_token_regex.search(encoding.tokens[first.token_ix]) is not None:
                css_classes.append('special-token')
                data_items['stok'] = encoding.tokens[first.token_ix]
        else:
            css_classes.append('non-token')
        css = f'class="{' '.join(css_classes)}"'
        data = ''
        for key, val in data_items.items():
            data += f' data-{key}="{val}"'
        return f'<span {css} {data} >{span_text}</span>'

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

    @staticmethod
    def __make_anno_map(text: str, annotations: AnnotationList) -> PartialIntList:
        """
        Args:
            text (:obj:`str`):
                The raw text we want to align to

            annotations (:obj:`AnnotationList`):
                A (possibly empty) list of annotations

        Returns:
            A list of  length len(text) whose entry at index i is None if there is no annotation on
            charachter i or k, the index of the annotation that covers index i where k is with
            respect to the list of annotations
        """
        annotation_map = [None] * len(text)
        for anno_ix, a in enumerate(annotations):
            for i in range(a.start, a.end):
                annotation_map[i] = anno_ix
        return annotation_map

    @staticmethod
    def __make_char_states(text: str, encoding: Encoding, annotations: AnnotationList) -> List[CharState]:
        """
        For each character in the original text, we emit a tuple representing it's "state":

            * which token_ix it corresponds to
            * which word_ix it corresponds to
            * which annotation_ix it corresponds to

        Args:
            text (:obj:`str`):
                The raw text we want to align to

            annotations (:obj:`List[Annotation]`):
                A (possibly empty) list of annotations

            encoding: (:class:`~tokenizers.Encoding`):
                The encoding returned from the tokenizer

        Returns:
            :obj:`List[CharState]`: A list of CharStates, indicating for each char in the text what
            it's state is
        """
        annotation_map = EncodingVisualizer.__make_anno_map(text, annotations)
        char_states: List[CharState] = [CharState(char_ix) for char_ix in range(len(text))]
        for token_ix, token in enumerate(encoding.tokens):
            offsets = encoding.token_to_chars(token_ix)
            if offsets is not None:
                start, end = offsets
                for i in range(start, end):
                    char_states[i].tokens.append(token_ix)
        for char_ix, anno_ix in enumerate(annotation_map):
            char_states[char_ix].anno_ix = anno_ix
        return char_states