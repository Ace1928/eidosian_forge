import datetime
from dataclasses import dataclass
from functools import wraps
from django.contrib.sites.shortcuts import get_current_site
from django.core.paginator import EmptyPage, PageNotAnInteger
from django.http import Http404
from django.template.response import TemplateResponse
from django.urls import reverse
from django.utils import timezone
from django.utils.http import http_date
def _get_latest_lastmod(current_lastmod, new_lastmod):
    """
    Returns the latest `lastmod` where `lastmod` can be either a date or a
    datetime.
    """
    if not isinstance(new_lastmod, datetime.datetime):
        new_lastmod = datetime.datetime.combine(new_lastmod, datetime.time.min)
    if timezone.is_naive(new_lastmod):
        new_lastmod = timezone.make_aware(new_lastmod, datetime.timezone.utc)
    return new_lastmod if current_lastmod is None else max(current_lastmod, new_lastmod)