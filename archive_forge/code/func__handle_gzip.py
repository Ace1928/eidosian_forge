import logging
import os.path
def _handle_gzip(file_obj, mode):
    import gzip
    result = gzip.GzipFile(fileobj=file_obj, mode=mode)
    tweak_close(result, file_obj)
    return result