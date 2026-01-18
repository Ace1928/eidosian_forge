@staticmethod
def _build_message(response, content, url):
    if isinstance(content, bytes):
        content = content.decode('ascii', 'replace')
    return 'HttpError accessing <%s>: response: <%s>, content <%s>' % (url, response, content)