from sympy.utilities.exceptions import sympy_deprecation_warning
from .facts import FactRules, FactKB
from .sympify import sympify
from sympy.core.random import _assumptions_shuffle as shuffle
from sympy.core.assumptions_generated import generated_assumptions as _assumptions
def _ask(fact, obj):
    """
    Find the truth value for a property of an object.

    This function is called when a request is made to see what a fact
    value is.

    For this we use several techniques:

    First, the fact-evaluation function is tried, if it exists (for
    example _eval_is_integer). Then we try related facts. For example

        rational   -->   integer

    another example is joined rule:

        integer & !odd  --> even

    so in the latter case if we are looking at what 'even' value is,
    'integer' and 'odd' facts will be asked.

    In all cases, when we settle on some fact value, its implications are
    deduced, and the result is cached in ._assumptions.
    """
    assumptions = obj._assumptions
    handler_map = obj._prop_handler
    facts_to_check = [fact]
    facts_queued = {fact}
    for fact_i in facts_to_check:
        if fact_i in assumptions:
            continue
        fact_i_value = None
        handler_i = handler_map.get(fact_i)
        if handler_i is not None:
            fact_i_value = handler_i(obj)
        if fact_i_value is not None:
            assumptions.deduce_all_facts(((fact_i, fact_i_value),))
        fact_value = assumptions.get(fact)
        if fact_value is not None:
            return fact_value
        new_facts_to_check = list(_assume_rules.prereq[fact_i] - facts_queued)
        shuffle(new_facts_to_check)
        facts_to_check.extend(new_facts_to_check)
        facts_queued.update(new_facts_to_check)
    if fact in assumptions:
        return assumptions[fact]
    assumptions._tell(fact, None)
    return None