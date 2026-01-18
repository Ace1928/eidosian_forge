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
@dataclass
class SitemapIndexItem:
    location: str
    last_mod: bool = None