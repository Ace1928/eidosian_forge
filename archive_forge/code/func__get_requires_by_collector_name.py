from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
import platform
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts import timeout
def _get_requires_by_collector_name(collector_name, all_fact_subsets):
    required_facts = set()
    try:
        collector_classes = all_fact_subsets[collector_name]
    except KeyError:
        raise CollectorNotFoundError('Fact collector "%s" not found' % collector_name)
    for collector_class in collector_classes:
        required_facts.update(collector_class.required_facts)
    return required_facts