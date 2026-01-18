def _is_appendable(file_like):
    if file_like.seekable() and file_like.tell() != 0:
        if '<stdout>' == getattr(file_like, 'name', ''):
            return False
        if file_like.readable():
            return True
        else:
            raise ValueError('When appending to an avro file you must use the ' + "'a+' mode, not just 'a'")
    else:
        return False