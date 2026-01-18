from urllib.parse import urlparse
from urllib.request import url2pathname
from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib.staticfiles import utils
from django.contrib.staticfiles.views import serve
from django.core.handlers.asgi import ASGIHandler
from django.core.handlers.exception import response_for_exception
from django.core.handlers.wsgi import WSGIHandler, get_path_info
from django.http import Http404
class ASGIStaticFilesHandler(StaticFilesHandlerMixin, ASGIHandler):
    """
    ASGI application which wraps another and intercepts requests for static
    files, passing them off to Django's static file serving.
    """

    def __init__(self, application):
        self.application = application
        self.base_url = urlparse(self.get_base_url())

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http' and self._should_handle(scope['path']):
            return await super().__call__(scope, receive, send)
        return await self.application(scope, receive, send)

    async def get_response_async(self, request):
        response = await super().get_response_async(request)
        response._resource_closers.append(request.close)
        if response.streaming and (not response.is_async):
            _iterator = response.streaming_content

            async def awrapper():
                for part in await sync_to_async(list)(_iterator):
                    yield part
            response.streaming_content = awrapper()
        return response