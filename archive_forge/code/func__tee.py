import io
def _tee(response, callback, chunksize, decode_content):
    for chunk in response.raw.stream(amt=chunksize, decode_content=decode_content):
        callback(chunk)
        yield chunk