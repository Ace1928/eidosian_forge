from functools import wraps
from asgiref.sync import iscoroutinefunction
from django.middleware.csrf import CsrfViewMiddleware, get_token
from django.utils.decorators import decorator_from_middleware
Mark a view function as being exempt from the CSRF view protection.