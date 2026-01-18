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
Get our DeltaGenerator.