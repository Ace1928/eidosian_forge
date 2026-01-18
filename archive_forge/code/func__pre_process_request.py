import datetime
from functools import wraps
from asgiref.sync import iscoroutinefunction
from django.http import HttpResponseNotAllowed
from django.middleware.http import ConditionalGetMiddleware
from django.utils import timezone
from django.utils.cache import get_conditional_response
from django.utils.decorators import decorator_from_middleware
from django.utils.http import http_date, quote_etag
from django.utils.log import log_response
def _pre_process_request(request, *args, **kwargs):
    res_last_modified = None
    if last_modified_func:
        if (dt := last_modified_func(request, *args, **kwargs)):
            if not timezone.is_aware(dt):
                dt = timezone.make_aware(dt, datetime.timezone.utc)
            res_last_modified = int(dt.timestamp())
    res_etag = etag_func(request, *args, **kwargs) if etag_func else None
    res_etag = quote_etag(res_etag) if res_etag is not None else None
    response = get_conditional_response(request, etag=res_etag, last_modified=res_last_modified)
    return (response, res_etag, res_last_modified)