from collections import Counter, defaultdict
from functools import partial, reduce
from itertools import chain
from operator import attrgetter, or_
from django.db import IntegrityError, connections, models, transaction
from django.db.models import query_utils, signals, sql
def _has_signal_listeners(self, model):
    return signals.pre_delete.has_listeners(model) or signals.post_delete.has_listeners(model)