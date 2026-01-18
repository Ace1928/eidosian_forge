from collections import Counter, defaultdict
from functools import partial, reduce
from itertools import chain
from operator import attrgetter, or_
from django.db import IntegrityError, connections, models, transaction
from django.db.models import query_utils, signals, sql
class ProtectedError(IntegrityError):

    def __init__(self, msg, protected_objects):
        self.protected_objects = protected_objects
        super().__init__(msg, protected_objects)