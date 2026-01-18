import io
import ntpath
import base64
def _io_to_str(data_io, writer, **kwargs):
    data_io_close = data_io.close
    data_io.close = lambda: None
    writer(data_io, **kwargs)
    data_value = data_io.getvalue()
    data_io_close()
    return data_value