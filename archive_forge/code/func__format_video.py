from __future__ import annotations
import tempfile
import warnings
from pathlib import Path
from typing import Any, Callable, Literal, Optional
from gradio_client import file
from gradio_client import utils as client_utils
from gradio_client.documentation import document
import gradio as gr
from gradio import processing_utils, utils, wasm_utils
from gradio.components.base import Component
from gradio.data_classes import FileData, GradioModel
from gradio.events import Events
def _format_video(self, video: str | Path | None) -> FileData | None:
    """
        Processes a video to ensure that it is in the correct format.
        """
    if video is None:
        return None
    video = str(video)
    returned_format = video.split('.')[-1].lower()
    if self.format is None or returned_format == self.format:
        conversion_needed = False
    else:
        conversion_needed = True
    is_url = client_utils.is_http_url_like(video)
    if is_url and (not conversion_needed):
        return FileData(path=video)
    if is_url:
        video = processing_utils.save_url_to_cache(video, cache_dir=self.GRADIO_CACHE)
    if processing_utils.ffmpeg_installed() and (not processing_utils.video_is_playable(video)):
        warnings.warn('Video does not have browser-compatible container or codec. Converting to mp4')
        video = processing_utils.convert_video_to_playable_mp4(video)
    returned_format = utils.get_extension_from_file_path_or_url(video).lower()
    if self.format is not None and returned_format != self.format:
        if wasm_utils.IS_WASM:
            raise wasm_utils.WasmUnsupportedError('Returning a video in a different format is not supported in the Wasm mode.')
        output_file_name = video[0:video.rindex('.') + 1] + self.format
        ff = FFmpeg(inputs={video: None}, outputs={output_file_name: None}, global_options='-y')
        ff.run()
        video = output_file_name
    return FileData(path=video, orig_name=Path(video).name)