from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from textwrap import dedent
from typing import (
from typing_extensions import TypeAlias
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.errors import StreamlitAPIException
from streamlit.proto.DateInput_pb2 import DateInput as DateInputProto
from streamlit.proto.TimeInput_pb2 import TimeInput as TimeInputProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id
from streamlit.time_util import adjust_years
from streamlit.type_util import Key, LabelVisibility, maybe_raise_label_warnings, to_key
class TimeWidgetsMixin:

    @overload
    def time_input(self, label: str, value: time | datetime | Literal['now']='now', key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, disabled: bool=False, label_visibility: LabelVisibility='visible', step: int | timedelta=timedelta(minutes=DEFAULT_STEP_MINUTES)) -> time:
        pass

    @overload
    def time_input(self, label: str, value: None=None, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, disabled: bool=False, label_visibility: LabelVisibility='visible', step: int | timedelta=timedelta(minutes=DEFAULT_STEP_MINUTES)) -> time | None:
        pass

    @gather_metrics('time_input')
    def time_input(self, label: str, value: time | datetime | Literal['now'] | None='now', key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, disabled: bool=False, label_visibility: LabelVisibility='visible', step: int | timedelta=timedelta(minutes=DEFAULT_STEP_MINUTES)) -> time | None:
        """Display a time input widget.

        Parameters
        ----------
        label : str
            A short label explaining to the user what this time input is for.
            The label can optionally contain Markdown and supports the following
            elements: Bold, Italics, Strikethroughs, Inline Code, Emojis, and Links.

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

            For accessibility reasons, you should never set an empty label (label="")
            but hide it with label_visibility if needed. In the future, we may disallow
            empty labels by raising an exception.
        value : datetime.time/datetime.datetime, "now" or None
            The value of this widget when it first renders. This will be
            cast to str internally. If ``None``, will initialize empty and
            return ``None`` until the user selects a time. If "now" (default),
            will initialize with the current time.
        key : str or int
            An optional string or integer to use as the unique key for the widget.
            If this is omitted, a key will be generated for the widget
            based on its content. Multiple widgets of the same type may
            not share the same key.
        help : str
            An optional tooltip that gets displayed next to the input.
        on_change : callable
            An optional callback invoked when this time_input's value changes.
        args : tuple
            An optional tuple of args to pass to the callback.
        kwargs : dict
            An optional dict of kwargs to pass to the callback.
        disabled : bool
            An optional boolean, which disables the time input if set to True.
            The default is False.
        label_visibility : "visible", "hidden", or "collapsed"
            The visibility of the label. If "hidden", the label doesn't show but there
            is still empty space for it above the widget (equivalent to label="").
            If "collapsed", both the label and the space are removed. Default is
            "visible".
        step : int or timedelta
            The stepping interval in seconds. Defaults to 900, i.e. 15 minutes.
            You can also pass a datetime.timedelta object.

        Returns
        -------
        datetime.time or None
            The current value of the time input widget or ``None`` if no time has been
            selected.

        Example
        -------
        >>> import datetime
        >>> import streamlit as st
        >>>
        >>> t = st.time_input('Set an alarm for', datetime.time(8, 45))
        >>> st.write('Alarm is set for', t)

        .. output::
           https://doc-time-input.streamlit.app/
           height: 260px

        To initialize an empty time input, use ``None`` as the value:

        >>> import datetime
        >>> import streamlit as st
        >>>
        >>> t = st.time_input('Set an alarm for', value=None)
        >>> st.write('Alarm is set for', t)

        .. output::
           https://doc-time-input-empty.streamlit.app/
           height: 260px

        """
        ctx = get_script_run_ctx()
        return self._time_input(label=label, value=value, key=key, help=help, on_change=on_change, args=args, kwargs=kwargs, disabled=disabled, label_visibility=label_visibility, step=step, ctx=ctx)

    def _time_input(self, label: str, value: time | datetime | Literal['now'] | None='now', key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, disabled: bool=False, label_visibility: LabelVisibility='visible', step: int | timedelta=timedelta(minutes=DEFAULT_STEP_MINUTES), ctx: ScriptRunContext | None=None) -> time | None:
        key = to_key(key)
        check_callback_rules(self.dg, on_change)
        check_session_state_rules(default_value=value if value != 'now' else None, key=key)
        maybe_raise_label_warnings(label, label_visibility)
        parsed_time: time | None
        if value is None:
            parsed_time = None
        elif value == 'now':
            parsed_time = datetime.now().time().replace(second=0, microsecond=0)
        elif isinstance(value, datetime):
            parsed_time = value.time().replace(second=0, microsecond=0)
        elif isinstance(value, time):
            parsed_time = value
        else:
            raise StreamlitAPIException('The type of value should be one of datetime, time or None')
        id = compute_widget_id('time_input', user_key=key, label=label, value=parsed_time if isinstance(value, (datetime, time)) else value, key=key, help=help, step=step, form_id=current_form_id(self.dg), page=ctx.page_script_hash if ctx else None)
        del value
        time_input_proto = TimeInputProto()
        time_input_proto.id = id
        time_input_proto.label = label
        if parsed_time is not None:
            time_input_proto.default = time.strftime(parsed_time, '%H:%M')
        time_input_proto.form_id = current_form_id(self.dg)
        if not isinstance(step, (int, timedelta)):
            raise StreamlitAPIException(f'`step` can only be `int` or `timedelta` but {type(step)} is provided.')
        if isinstance(step, timedelta):
            step = step.seconds
        if step < 60 or step > timedelta(hours=23).seconds:
            raise StreamlitAPIException(f'`step` must be between 60 seconds and 23 hours but is currently set to {step} seconds.')
        time_input_proto.step = step
        time_input_proto.disabled = disabled
        time_input_proto.label_visibility.value = get_label_visibility_proto_value(label_visibility)
        if help is not None:
            time_input_proto.help = dedent(help)
        serde = TimeInputSerde(parsed_time)
        widget_state = register_widget('time_input', time_input_proto, user_key=key, on_change_handler=on_change, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
        if widget_state.value_changed:
            if (serialized_value := serde.serialize(widget_state.value)) is not None:
                time_input_proto.value = serialized_value
            time_input_proto.set_value = True
        self.dg._enqueue('time_input', time_input_proto)
        return widget_state.value

    @gather_metrics('date_input')
    def date_input(self, label: str, value: DateValue | Literal['today']='default_value_today', min_value: SingleDateValue=None, max_value: SingleDateValue=None, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, format: str='YYYY/MM/DD', disabled: bool=False, label_visibility: LabelVisibility='visible') -> DateWidgetReturn:
        """Display a date input widget.

        Parameters
        ----------
        label : str
            A short label explaining to the user what this date input is for.
            The label can optionally contain Markdown and supports the following
            elements: Bold, Italics, Strikethroughs, Inline Code, Emojis, and Links.

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

            For accessibility reasons, you should never set an empty label (label="")
            but hide it with label_visibility if needed. In the future, we may disallow
            empty labels by raising an exception.
        value : datetime.date or datetime.datetime or list/tuple of datetime.date or datetime.datetime, "today", or None
            The value of this widget when it first renders. If a list/tuple with
            0 to 2 date/datetime values is provided, the datepicker will allow
            users to provide a range. If ``None``, will initialize empty and
            return ``None`` until the user provides input. If "today" (default),
            will initialize with today as a single-date picker.
        min_value : datetime.date or datetime.datetime
            The minimum selectable date. If value is a date, defaults to value - 10 years.
            If value is the interval [start, end], defaults to start - 10 years.
        max_value : datetime.date or datetime.datetime
            The maximum selectable date. If value is a date, defaults to value + 10 years.
            If value is the interval [start, end], defaults to end + 10 years.
        key : str or int
            An optional string or integer to use as the unique key for the widget.
            If this is omitted, a key will be generated for the widget
            based on its content. Multiple widgets of the same type may
            not share the same key.
        help : str
            An optional tooltip that gets displayed next to the input.
        on_change : callable
            An optional callback invoked when this date_input's value changes.
        args : tuple
            An optional tuple of args to pass to the callback.
        kwargs : dict
            An optional dict of kwargs to pass to the callback.
        format : str
            A format string controlling how the interface should display dates.
            Supports “YYYY/MM/DD” (default), “DD/MM/YYYY”, or “MM/DD/YYYY”.
            You may also use a period (.) or hyphen (-) as separators.
        disabled : bool
            An optional boolean, which disables the date input if set to True.
            The default is False.
        label_visibility : "visible", "hidden", or "collapsed"
            The visibility of the label. If "hidden", the label doesn't show but there
            is still empty space for it above the widget (equivalent to label="").
            If "collapsed", both the label and the space are removed. Default is
            "visible".


        Returns
        -------
        datetime.date or a tuple with 0-2 dates or None
            The current value of the date input widget or ``None`` if no date has been
            selected.

        Examples
        --------
        >>> import datetime
        >>> import streamlit as st
        >>>
        >>> d = st.date_input("When's your birthday", datetime.date(2019, 7, 6))
        >>> st.write('Your birthday is:', d)

        .. output::
           https://doc-date-input.streamlit.app/
           height: 380px

        >>> import datetime
        >>> import streamlit as st
        >>>
        >>> today = datetime.datetime.now()
        >>> next_year = today.year + 1
        >>> jan_1 = datetime.date(next_year, 1, 1)
        >>> dec_31 = datetime.date(next_year, 12, 31)
        >>>
        >>> d = st.date_input(
        ...     "Select your vacation for next year",
        ...     (jan_1, datetime.date(next_year, 1, 7)),
        ...     jan_1,
        ...     dec_31,
        ...     format="MM.DD.YYYY",
        ... )
        >>> d

        .. output::
           https://doc-date-input1.streamlit.app/
           height: 380px

        To initialize an empty date input, use ``None`` as the value:

        >>> import datetime
        >>> import streamlit as st
        >>>
        >>> d = st.date_input("When's your birthday", value=None)
        >>> st.write('Your birthday is:', d)

        .. output::
           https://doc-date-input-empty.streamlit.app/
           height: 380px

        """
        ctx = get_script_run_ctx()
        return self._date_input(label=label, value=value, min_value=min_value, max_value=max_value, key=key, help=help, on_change=on_change, args=args, kwargs=kwargs, disabled=disabled, label_visibility=label_visibility, format=format, ctx=ctx)

    def _date_input(self, label: str, value: DateValue | Literal['today'] | Literal['default_value_today']='default_value_today', min_value: SingleDateValue=None, max_value: SingleDateValue=None, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, format: str='YYYY/MM/DD', disabled: bool=False, label_visibility: LabelVisibility='visible', ctx: ScriptRunContext | None=None) -> DateWidgetReturn:
        key = to_key(key)
        check_callback_rules(self.dg, on_change)
        check_session_state_rules(default_value=value if value != 'default_value_today' else None, key=key)
        maybe_raise_label_warnings(label, label_visibility)

        def parse_date_deterministic(v: SingleDateValue | Literal['today'] | Literal['default_value_today']) -> str | None:
            if isinstance(v, datetime):
                return date.strftime(v.date(), '%Y/%m/%d')
            elif isinstance(v, date):
                return date.strftime(v, '%Y/%m/%d')
            return None
        parsed_min_date = parse_date_deterministic(min_value)
        parsed_max_date = parse_date_deterministic(max_value)
        parsed: str | None | list[str | None]
        if value == 'today' or value == 'default_value_today' or value is None:
            parsed = None
        elif isinstance(value, (datetime, date)):
            parsed = parse_date_deterministic(value)
        else:
            parsed = [parse_date_deterministic(cast(SingleDateValue, v)) for v in value]
        id = compute_widget_id('date_input', user_key=key, label=label, value=parsed, min_value=parsed_min_date, max_value=parsed_max_date, key=key, help=help, format=format, form_id=current_form_id(self.dg), page=ctx.page_script_hash if ctx else None)
        if not bool(ALLOWED_DATE_FORMATS.match(format)):
            raise StreamlitAPIException(f'The provided format (`{format}`) is not valid. DateInput format should be one of `YYYY/MM/DD`, `DD/MM/YYYY`, or `MM/DD/YYYY` and can also use a period (.) or hyphen (-) as separators.')
        parsed_values = _DateInputValues.from_raw_values(value=value, min_value=min_value, max_value=max_value)
        if value == 'default_value_today':
            session_state = get_session_state().filtered_state
            if key is not None and key in session_state:
                state_value = session_state[key]
                parsed_values = _DateInputValues.from_raw_values(value=state_value, min_value=min_value, max_value=max_value)
        del value, min_value, max_value
        date_input_proto = DateInputProto()
        date_input_proto.id = id
        date_input_proto.is_range = parsed_values.is_range
        date_input_proto.disabled = disabled
        date_input_proto.label_visibility.value = get_label_visibility_proto_value(label_visibility)
        date_input_proto.format = format
        date_input_proto.label = label
        if parsed_values.value is None:
            date_input_proto.default[:] = []
        else:
            date_input_proto.default[:] = [date.strftime(v, '%Y/%m/%d') for v in parsed_values.value]
        date_input_proto.min = date.strftime(parsed_values.min, '%Y/%m/%d')
        date_input_proto.max = date.strftime(parsed_values.max, '%Y/%m/%d')
        date_input_proto.form_id = current_form_id(self.dg)
        if help is not None:
            date_input_proto.help = dedent(help)
        serde = DateInputSerde(parsed_values)
        widget_state = register_widget('date_input', date_input_proto, user_key=key, on_change_handler=on_change, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
        if widget_state.value_changed:
            date_input_proto.value[:] = serde.serialize(widget_state.value)
            date_input_proto.set_value = True
        self.dg._enqueue('date_input', date_input_proto)
        return widget_state.value

    @property
    def dg(self) -> DeltaGenerator:
        """Get our DeltaGenerator."""
        return cast('DeltaGenerator', self)