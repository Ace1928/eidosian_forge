from collections import Counter, defaultdict
from functools import partial, reduce
from itertools import chain
from operator import attrgetter, or_
from django.db import IntegrityError, connections, models, transaction
from django.db.models import query_utils, signals, sql
def CASCADE(collector, field, sub_objs, using):
    collector.collect(sub_objs, source=field.remote_field.model, source_attr=field.name, nullable=field.null, fail_on_restricted=False)
    if field.null and (not connections[using].features.can_defer_constraint_checks):
        collector.add_field_update(field, None, sub_objs)