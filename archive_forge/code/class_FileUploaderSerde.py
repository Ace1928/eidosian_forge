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
@dataclass
class FileUploaderSerde:
    accept_multiple_files: bool

    def deserialize(self, ui_value: FileUploaderStateProto | None, widget_id: str) -> SomeUploadedFiles:
        upload_files = _get_upload_files(ui_value)
        if len(upload_files) == 0:
            return_value: SomeUploadedFiles = [] if self.accept_multiple_files else None
        else:
            return_value = upload_files if self.accept_multiple_files else upload_files[0]
        return return_value

    def serialize(self, files: SomeUploadedFiles) -> FileUploaderStateProto:
        state_proto = FileUploaderStateProto()
        if not files:
            return state_proto
        elif not isinstance(files, list):
            files = [files]
        for f in files:
            if isinstance(f, DeletedFile):
                continue
            file_info: UploadedFileInfoProto = state_proto.uploaded_file_info.add()
            file_info.file_id = f.file_id
            file_info.name = f.name
            file_info.size = f.size
            file_info.file_urls.CopyFrom(f._file_urls)
        return state_proto