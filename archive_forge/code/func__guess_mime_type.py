import os
import pathlib
from mimetypes import guess_type
def _guess_mime_type(file_path):
    filename = pathlib.Path(file_path).name
    extension = os.path.splitext(filename)[-1].replace('.', '')
    if extension == '':
        extension = filename
    if extension in get_text_extensions():
        return 'text/plain'
    mime_type, _ = guess_type(filename)
    if not mime_type:
        return 'application/octet-stream'
    return mime_type