from __future__ import annotations
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, Literal, Union, cast
from typing_extensions import TypeAlias
from streamlit.elements.utils import get_label_visibility_proto_value
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Metric_pb2 import Metric as MetricProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.string_util import clean_text
from streamlit.type_util import LabelVisibility, maybe_raise_label_warnings
class MetricMixin:

    @gather_metrics('metric')
    def metric(self, label: str, value: Value, delta: Delta=None, delta_color: DeltaColor='normal', help: str | None=None, label_visibility: LabelVisibility='visible') -> DeltaGenerator:
        """Display a metric in big bold font, with an optional indicator of how the metric changed.

        Tip: If you want to display a large number, it may be a good idea to
        shorten it using packages like `millify <https://github.com/azaitsev/millify>`_
        or `numerize <https://github.com/davidsa03/numerize>`_. E.g. ``1234`` can be
        displayed as ``1.2k`` using ``st.metric("Short number", millify(1234))``.

        Parameters
        ----------
        label : str
            The header or title for the metric. The label can optionally contain
            Markdown and supports the following elements: Bold, Italics,
            Strikethroughs, Inline Code, Emojis, and Links.

            This also supports:

            * Emoji shortcodes, such as ``:+1:``  and ``:sunglasses:``.
              For a list of all supported codes,
              see https://share.streamlit.io/streamlit/emoji-shortcodes.

            * LaTeX expressions, by wrapping them in "$" or "$$" (the "$$"
              must be on their own lines). Supported LaTeX functions are listed
              at https://katex.org/docs/supported.html.

            * Colored text, using the syntax ``:color[text to be colored]``,
              where ``color`` needs to be replaced with any of the following
              supported colors: blue, green, orange, red, violet, gray/grey, rainbow.

            Unsupported elements are unwrapped so only their children (text contents) render.
            Display unsupported elements as literal characters by
            backslash-escaping them. E.g. ``1\\. Not an ordered list``.
        value : int, float, str, or None
             Value of the metric. None is rendered as a long dash.
        delta : int, float, str, or None
            Indicator of how the metric changed, rendered with an arrow below
            the metric. If delta is negative (int/float) or starts with a minus
            sign (str), the arrow points down and the text is red; else the
            arrow points up and the text is green. If None (default), no delta
            indicator is shown.
        delta_color : "normal", "inverse", or "off"
             If "normal" (default), the delta indicator is shown as described
             above. If "inverse", it is red when positive and green when
             negative. This is useful when a negative change is considered
             good, e.g. if cost decreased. If "off", delta is  shown in gray
             regardless of its value.
        help : str
            An optional tooltip that gets displayed next to the metric label.
        label_visibility : "visible", "hidden", or "collapsed"
            The visibility of the label. If "hidden", the label doesn't show but there
            is still empty space for it (equivalent to label="").
            If "collapsed", both the label and the space are removed. Default is
            "visible".

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> st.metric(label="Temperature", value="70 °F", delta="1.2 °F")

        .. output::
            https://doc-metric-example1.streamlit.app/
            height: 210px

        ``st.metric`` looks especially nice in combination with ``st.columns``:

        >>> import streamlit as st
        >>>
        >>> col1, col2, col3 = st.columns(3)
        >>> col1.metric("Temperature", "70 °F", "1.2 °F")
        >>> col2.metric("Wind", "9 mph", "-8%")
        >>> col3.metric("Humidity", "86%", "4%")

        .. output::
            https://doc-metric-example2.streamlit.app/
            height: 210px

        The delta indicator color can also be inverted or turned off:

        >>> import streamlit as st
        >>>
        >>> st.metric(label="Gas price", value=4, delta=-0.5,
        ...     delta_color="inverse")
        >>>
        >>> st.metric(label="Active developers", value=123, delta=123,
        ...     delta_color="off")

        .. output::
            https://doc-metric-example3.streamlit.app/
            height: 320px

        """
        maybe_raise_label_warnings(label, label_visibility)
        metric_proto = MetricProto()
        metric_proto.body = _parse_value(value)
        metric_proto.label = _parse_label(label)
        metric_proto.delta = _parse_delta(delta)
        if help is not None:
            metric_proto.help = dedent(help)
        color_and_direction = _determine_delta_color_and_direction(cast(DeltaColor, clean_text(delta_color)), delta)
        metric_proto.color = color_and_direction.color
        metric_proto.direction = color_and_direction.direction
        metric_proto.label_visibility.value = get_label_visibility_proto_value(label_visibility)
        return self.dg._enqueue('metric', metric_proto)

    @property
    def dg(self) -> DeltaGenerator:
        return cast('DeltaGenerator', self)