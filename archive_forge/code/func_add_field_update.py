from collections import Counter, defaultdict
from functools import partial, reduce
from itertools import chain
from operator import attrgetter, or_
from django.db import IntegrityError, connections, models, transaction
from django.db.models import query_utils, signals, sql
def add_field_update(self, field, value, objs):
    """
        Schedule a field update. 'objs' must be a homogeneous iterable
        collection of model instances (e.g. a QuerySet).
        """
    self.field_updates[field, value].append(objs)