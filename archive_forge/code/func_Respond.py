import gzip
import io
import json
import re
import struct
import time
import wsgiref.handlers
import werkzeug
from tensorboard.backend import json_util
def Respond(request, content, content_type, code=200, expires=0, content_encoding=None, encoding='utf-8', csp_scripts_sha256s=None, headers=None):
    """Construct a werkzeug Response.

    Responses are transmitted to the browser with compression if: a) the browser
    supports it; b) it's sane to compress the content_type in question; and c)
    the content isn't already compressed, as indicated by the content_encoding
    parameter.

    Browser and proxy caching is completely disabled by default. If the expires
    parameter is greater than zero then the response will be able to be cached by
    the browser for that many seconds; however, proxies are still forbidden from
    caching so that developers can bypass the cache with Ctrl+Shift+R.

    For textual content that isn't JSON, the encoding parameter is used as the
    transmission charset which is automatically appended to the Content-Type
    header. That is unless of course the content_type parameter contains a
    charset parameter. If the two disagree, the characters in content will be
    transcoded to the latter.

    If content_type declares a JSON media type, then content MAY be a dict, list,
    tuple, or set, in which case this function has an implicit composition with
    json_util.Cleanse and json.dumps. The encoding parameter is used to decode
    byte strings within the JSON object; therefore transmitting binary data
    within JSON is not permitted. JSON is transmitted as ASCII unless the
    content_type parameter explicitly defines a charset parameter, in which case
    the serialized JSON bytes will use that instead of escape sequences.

    Args:
      request: A werkzeug Request object. Used mostly to check the
        Accept-Encoding header.
      content: Payload data as byte string, unicode string, or maybe JSON.
      content_type: Media type and optionally an output charset.
      code: Numeric HTTP status code to use.
      expires: Second duration for browser caching.
      content_encoding: Encoding if content is already encoded, e.g. 'gzip'.
      encoding: Input charset if content parameter has byte strings.
      csp_scripts_sha256s: List of base64 serialized sha256 of whitelisted script
        elements for script-src of the Content-Security-Policy. If it is None, the
        HTML will disallow any script to execute. It is only be used when the
        content_type is text/html.
      headers: Any additional headers to include on the response, as a
        list of key-value tuples: e.g., `[("Allow", "GET")]`. In case of
        conflict, these may be overridden with headers added by this function.

    Returns:
      A werkzeug Response object (a WSGI application).
    """
    mimetype = _EXTRACT_MIMETYPE_PATTERN.search(content_type).group(0)
    charset_match = _EXTRACT_CHARSET_PATTERN.search(content_type)
    charset = charset_match.group(1) if charset_match else encoding
    textual = charset_match or mimetype in _TEXTUAL_MIMETYPES
    if mimetype in _JSON_MIMETYPES and isinstance(content, (dict, list, set, tuple)):
        content = json.dumps(json_util.Cleanse(content, encoding), ensure_ascii=not charset_match)
    if charset != encoding and isinstance(content, bytes):
        content = content.decode(encoding)
    if isinstance(content, str):
        content = content.encode(charset)
    if textual and (not charset_match) and (mimetype not in _JSON_MIMETYPES):
        content_type += '; charset=' + charset
    gzip_accepted = _ALLOWS_GZIP_PATTERN.search(request.headers.get('Accept-Encoding', ''))
    if textual and (not content_encoding) and gzip_accepted:
        out = io.BytesIO()
        with gzip.GzipFile(fileobj=out, mode='wb', compresslevel=3, mtime=0) as f:
            f.write(content)
        content = out.getvalue()
        content_encoding = 'gzip'
    content_length = len(content)
    direct_passthrough = False
    if content_encoding == 'gzip' and (not gzip_accepted):
        gzip_file = gzip.GzipFile(fileobj=io.BytesIO(content), mode='rb')
        content_length = struct.unpack('<I', content[-4:])[0]
        content = werkzeug.wsgi.wrap_file(request.environ, gzip_file)
        content_encoding = None
        direct_passthrough = True
    headers = list(headers or [])
    headers.append(('Content-Length', str(content_length)))
    headers.append(('X-Content-Type-Options', 'nosniff'))
    if content_encoding:
        headers.append(('Content-Encoding', content_encoding))
    if expires > 0:
        e = wsgiref.handlers.format_date_time(time.time() + float(expires))
        headers.append(('Expires', e))
        headers.append(('Cache-Control', 'private, max-age=%d' % expires))
    else:
        headers.append(('Expires', '0'))
        headers.append(('Cache-Control', 'no-cache, must-revalidate'))
    if mimetype == _HTML_MIMETYPE:
        frags = _CSP_SCRIPT_DOMAINS_WHITELIST + ["'self'" if _CSP_SCRIPT_SELF else '', "'unsafe-eval'" if _CSP_SCRIPT_UNSAFE_EVAL else ''] + ["'sha256-{}'".format(sha256) for sha256 in csp_scripts_sha256s or []]
        script_srcs = _create_csp_string(*frags)
        csp_string = ';'.join(["default-src 'self'", 'font-src %s' % _create_csp_string("'self'", *_CSP_FONT_DOMAINS_WHITELIST), 'frame-src %s' % _create_csp_string("'self'", *_CSP_FRAME_DOMAINS_WHITELIST), 'img-src %s' % _create_csp_string("'self'", 'data:', 'blob:', *_CSP_IMG_DOMAINS_WHITELIST), "object-src 'none'", 'style-src %s' % _create_csp_string("'self'", 'https://www.gstatic.com', 'data:', "'unsafe-inline'", *_CSP_STYLE_DOMAINS_WHITELIST), 'connect-src %s' % _create_csp_string("'self'", *_CSP_CONNECT_DOMAINS_WHITELIST), 'script-src %s' % script_srcs])
        headers.append(('Content-Security-Policy', csp_string))
    if request.method == 'HEAD':
        content = None
    return werkzeug.wrappers.Response(response=content, status=code, headers=headers, content_type=content_type, direct_passthrough=direct_passthrough)