import os
import os.path
import struct
from io import BytesIO
from typing import BinaryIO, Tuple
from .jbig2 import JBIG2StreamReader, JBIG2StreamWriter
from .layout import LTImage
from .pdfcolor import LITERAL_DEVICE_CMYK
from .pdfcolor import LITERAL_DEVICE_GRAY
from .pdfcolor import LITERAL_DEVICE_RGB
from .pdftypes import (
def _save_jbig2(self, image: LTImage) -> str:
    """Save a JBIG2 encoded image"""
    name, path = self._create_unique_image_name(image, '.jb2')
    with open(path, 'wb') as fp:
        input_stream = BytesIO()
        global_streams = []
        filters = image.stream.get_filters()
        for filter_name, params in filters:
            if filter_name in LITERALS_JBIG2_DECODE:
                global_streams.append(params['JBIG2Globals'].resolve())
        if len(global_streams) > 1:
            msg = 'There should never be more than one JBIG2Globals associated with a JBIG2 embedded image'
            raise ValueError(msg)
        if len(global_streams) == 1:
            input_stream.write(global_streams[0].get_data().rstrip(b'\n'))
        input_stream.write(image.stream.get_data())
        input_stream.seek(0)
        reader = JBIG2StreamReader(input_stream)
        segments = reader.get_segments()
        writer = JBIG2StreamWriter(fp)
        writer.write_file(segments)
    return name