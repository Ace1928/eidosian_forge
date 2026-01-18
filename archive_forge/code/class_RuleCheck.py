import abc
import ast
import inspect
import stevedore
@register('rule')
class RuleCheck(Check):

    def __call__(self, target, creds, enforcer, current_rule=None):
        try:
            return _check(rule=enforcer.rules[self.match], target=target, creds=creds, enforcer=enforcer, current_rule=current_rule)
        except KeyError:
            return False