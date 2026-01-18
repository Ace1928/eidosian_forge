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
def _get_upload_files(widget_value: FileUploaderStateProto | None) -> list[UploadedFile | DeletedFile]:
    if widget_value is None:
        return []
    ctx = get_script_run_ctx()
    if ctx is None:
        return []
    uploaded_file_info = widget_value.uploaded_file_info
    if len(uploaded_file_info) == 0:
        return []
    file_recs_list = ctx.uploaded_file_mgr.get_files(session_id=ctx.session_id, file_ids=[f.file_id for f in uploaded_file_info])
    file_recs = {f.file_id: f for f in file_recs_list}
    collected_files: list[UploadedFile | DeletedFile] = []
    for f in uploaded_file_info:
        maybe_file_rec = file_recs.get(f.file_id)
        if maybe_file_rec is not None:
            uploaded_file = UploadedFile(maybe_file_rec, f.file_urls)
            collected_files.append(uploaded_file)
        else:
            collected_files.append(DeletedFile(f.file_id))
    return collected_files