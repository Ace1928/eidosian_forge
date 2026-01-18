from jmespath.compat import with_str_method
@with_str_method
class VariadictArityError(ArityError):

    def __str__(self):
        return 'Expected at least %s %s for function %s(), received %s' % (self.expected_arity, self._pluralize('argument', self.expected_arity), self.function_name, self.actual_arity)