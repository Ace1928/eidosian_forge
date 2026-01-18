from django.utils.cache import patch_vary_headers
from django.utils.deprecation import MiddlewareMixin
from django.utils.regex_helper import _lazy_re_compile
from django.utils.text import compress_sequence, compress_string
class GZipMiddleware(MiddlewareMixin):
    """
    Compress content if the browser allows gzip compression.
    Set the Vary header accordingly, so that caches will base their storage
    on the Accept-Encoding header.
    """
    max_random_bytes = 100

    def process_response(self, request, response):
        if not response.streaming and len(response.content) < 200:
            return response
        if response.has_header('Content-Encoding'):
            return response
        patch_vary_headers(response, ('Accept-Encoding',))
        ae = request.META.get('HTTP_ACCEPT_ENCODING', '')
        if not re_accepts_gzip.search(ae):
            return response
        if response.streaming:
            if response.is_async:
                orignal_iterator = response.streaming_content

                async def gzip_wrapper():
                    async for chunk in orignal_iterator:
                        yield compress_string(chunk, max_random_bytes=self.max_random_bytes)
                response.streaming_content = gzip_wrapper()
            else:
                response.streaming_content = compress_sequence(response.streaming_content, max_random_bytes=self.max_random_bytes)
            del response.headers['Content-Length']
        else:
            compressed_content = compress_string(response.content, max_random_bytes=self.max_random_bytes)
            if len(compressed_content) >= len(response.content):
                return response
            response.content = compressed_content
            response.headers['Content-Length'] = str(len(response.content))
        etag = response.get('ETag')
        if etag and etag.startswith('"'):
            response.headers['ETag'] = 'W/' + etag
        response.headers['Content-Encoding'] = 'gzip'
        return response