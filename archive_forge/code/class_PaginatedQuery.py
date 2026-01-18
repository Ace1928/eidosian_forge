import math
import sys
from flask import abort
from flask import render_template
from flask import request
from peewee import Database
from peewee import DoesNotExist
from peewee import Model
from peewee import Proxy
from peewee import SelectQuery
from playhouse.db_url import connect as db_url_connect
class PaginatedQuery(object):

    def __init__(self, query_or_model, paginate_by, page_var='page', page=None, check_bounds=False):
        self.paginate_by = paginate_by
        self.page_var = page_var
        self.page = page or None
        self.check_bounds = check_bounds
        if isinstance(query_or_model, SelectQuery):
            self.query = query_or_model
            self.model = self.query.model
        else:
            self.model = query_or_model
            self.query = self.model.select()

    def get_page(self):
        if self.page is not None:
            return self.page
        curr_page = request.args.get(self.page_var)
        if curr_page and curr_page.isdigit():
            return max(1, int(curr_page))
        return 1

    def get_page_count(self):
        if not hasattr(self, '_page_count'):
            self._page_count = int(math.ceil(float(self.query.count()) / self.paginate_by))
        return self._page_count

    def get_object_list(self):
        if self.check_bounds and self.get_page() > self.get_page_count():
            abort(404)
        return self.query.paginate(self.get_page(), self.paginate_by)

    def get_page_range(self, page, total, show=5):
        start = max(page - show // 2, 1)
        stop = min(start + show, total) + 1
        start = max(min(start, stop - show), 1)
        return list(range(start, stop)[:show])