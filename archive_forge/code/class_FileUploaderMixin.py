from __future__ import annotations
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, List, Literal, Sequence, Union, cast, overload
from typing_extensions import TypeAlias
from streamlit import config
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.proto.Common_pb2 import FileUploaderState as FileUploaderStateProto
from streamlit.proto.Common_pb2 import UploadedFileInfo as UploadedFileInfoProto
from streamlit.proto.FileUploader_pb2 import FileUploader as FileUploaderProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id
from streamlit.runtime.uploaded_file_manager import DeletedFile, UploadedFile
from streamlit.type_util import Key, LabelVisibility, maybe_raise_label_warnings, to_key
class FileUploaderMixin:

    @overload
    def file_uploader(self, label: str, type: str | Sequence[str] | None, accept_multiple_files: Literal[True], key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, disabled: bool=False, label_visibility: LabelVisibility='visible') -> list[UploadedFile] | None:
        ...

    @overload
    def file_uploader(self, label: str, type: str | Sequence[str] | None, accept_multiple_files: Literal[False]=False, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, disabled: bool=False, label_visibility: LabelVisibility='visible') -> UploadedFile | None:
        ...

    @overload
    def file_uploader(self, label: str, *, accept_multiple_files: Literal[True], type: str | Sequence[str] | None=None, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, disabled: bool=False, label_visibility: LabelVisibility='visible') -> list[UploadedFile] | None:
        ...

    @overload
    def file_uploader(self, label: str, *, accept_multiple_files: Literal[False]=False, type: str | Sequence[str] | None=None, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, disabled: bool=False, label_visibility: LabelVisibility='visible') -> UploadedFile | None:
        ...

    @gather_metrics('file_uploader')
    def file_uploader(self, label: str, type: str | Sequence[str] | None=None, accept_multiple_files: bool=False, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, disabled: bool=False, label_visibility: LabelVisibility='visible') -> UploadedFile | list[UploadedFile] | None:
        """Display a file uploader widget.
        By default, uploaded files are limited to 200MB. You can configure
        this using the `server.maxUploadSize` config option. For more info
        on how to set config options, see
        https://docs.streamlit.io/library/advanced-features/configuration#set-configuration-options

        Parameters
        ----------
        label : str
            A short label explaining to the user what this file uploader is for.
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

        type : str or list of str or None
            Array of allowed extensions. ['png', 'jpg']
            The default is None, which means all extensions are allowed.

        accept_multiple_files : bool
            If True, allows the user to upload multiple files at the same time,
            in which case the return value will be a list of files.
            Default: False

        key : str or int
            An optional string or integer to use as the unique key for the widget.
            If this is omitted, a key will be generated for the widget
            based on its content. Multiple widgets of the same type may
            not share the same key.

        help : str
            A tooltip that gets displayed next to the file uploader.

        on_change : callable
            An optional callback invoked when this file_uploader's value
            changes.

        args : tuple
            An optional tuple of args to pass to the callback.

        kwargs : dict
            An optional dict of kwargs to pass to the callback.

        disabled : bool
            An optional boolean, which disables the file uploader if set to
            True. The default is False. This argument can only be supplied by
            keyword.
        label_visibility : "visible", "hidden", or "collapsed"
            The visibility of the label. If "hidden", the label doesn't show but there
            is still empty space for it above the widget (equivalent to label="").
            If "collapsed", both the label and the space are removed. Default is
            "visible".

        Returns
        -------
        None or UploadedFile or list of UploadedFile
            - If accept_multiple_files is False, returns either None or
              an UploadedFile object.
            - If accept_multiple_files is True, returns a list with the
              uploaded files as UploadedFile objects. If no files were
              uploaded, returns an empty list.

            The UploadedFile class is a subclass of BytesIO, and therefore
            it is "file-like". This means you can pass them anywhere where
            a file is expected.

        Examples
        --------
        Insert a file uploader that accepts a single file at a time:

        >>> import streamlit as st
        >>> import pandas as pd
        >>> from io import StringIO
        >>>
        >>> uploaded_file = st.file_uploader("Choose a file")
        >>> if uploaded_file is not None:
        ...     # To read file as bytes:
        ...     bytes_data = uploaded_file.getvalue()
        ...     st.write(bytes_data)
        >>>
        ...     # To convert to a string based IO:
        ...     stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        ...     st.write(stringio)
        >>>
        ...     # To read file as string:
        ...     string_data = stringio.read()
        ...     st.write(string_data)
        >>>
        ...     # Can be used wherever a "file-like" object is accepted:
        ...     dataframe = pd.read_csv(uploaded_file)
        ...     st.write(dataframe)

        Insert a file uploader that accepts multiple files at a time:

        >>> import streamlit as st
        >>>
        >>> uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
        >>> for uploaded_file in uploaded_files:
        ...     bytes_data = uploaded_file.read()
        ...     st.write("filename:", uploaded_file.name)
        ...     st.write(bytes_data)

        .. output::
           https://doc-file-uploader.streamlit.app/
           height: 375px

        """
        ctx = get_script_run_ctx()
        return self._file_uploader(label=label, type=type, accept_multiple_files=accept_multiple_files, key=key, help=help, on_change=on_change, args=args, kwargs=kwargs, disabled=disabled, label_visibility=label_visibility, ctx=ctx)

    def _file_uploader(self, label: str, type: str | Sequence[str] | None=None, accept_multiple_files: bool=False, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, label_visibility: LabelVisibility='visible', disabled: bool=False, ctx: ScriptRunContext | None=None) -> UploadedFile | list[UploadedFile] | None:
        key = to_key(key)
        check_callback_rules(self.dg, on_change)
        check_session_state_rules(default_value=None, key=key, writes_allowed=False)
        maybe_raise_label_warnings(label, label_visibility)
        id = compute_widget_id('file_uploader', user_key=key, label=label, type=type, accept_multiple_files=accept_multiple_files, key=key, help=help, form_id=current_form_id(self.dg), page=ctx.page_script_hash if ctx else None)
        if type:
            if isinstance(type, str):
                type = [type]
            type = [file_type if file_type[0] == '.' else f'.{file_type}' for file_type in type]
            type = [t.lower() for t in type]
            for x, y in TYPE_PAIRS:
                if x in type and y not in type:
                    type.append(y)
                if y in type and x not in type:
                    type.append(x)
        file_uploader_proto = FileUploaderProto()
        file_uploader_proto.id = id
        file_uploader_proto.label = label
        file_uploader_proto.type[:] = type if type is not None else []
        file_uploader_proto.max_upload_size_mb = config.get_option('server.maxUploadSize')
        file_uploader_proto.multiple_files = accept_multiple_files
        file_uploader_proto.form_id = current_form_id(self.dg)
        file_uploader_proto.disabled = disabled
        file_uploader_proto.label_visibility.value = get_label_visibility_proto_value(label_visibility)
        if help is not None:
            file_uploader_proto.help = dedent(help)
        serde = FileUploaderSerde(accept_multiple_files)
        widget_state = register_widget('file_uploader', file_uploader_proto, user_key=key, on_change_handler=on_change, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
        self.dg._enqueue('file_uploader', file_uploader_proto)
        if isinstance(widget_state.value, DeletedFile):
            return None
        elif isinstance(widget_state.value, list):
            return [f for f in widget_state.value if not isinstance(f, DeletedFile)]
        return widget_state.value

    @property
    def dg(self) -> DeltaGenerator:
        """Get our DeltaGenerator."""
        return cast('DeltaGenerator', self)