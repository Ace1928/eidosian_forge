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